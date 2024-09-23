import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import numpy as np
from TaCo.tools.visualization import plot_co_occurence


class LogisticMLP(nn.Module):
    """Logistic regression or MLP depending on model_type."""

    def __init__(self, in_features, hidden_features, nb_classes, model_type):
        """Initialize model.
        
        Args:
            in_features: int, number of input features.
            hidden_features: int, number of hidden features.
            nb_classes: int, number of classes.
            model_type: str, either 'linear' or 'mlp'.
        """
        super().__init__()
        self.model_type = model_type
        if model_type == 'linear':
            self.fc1 = nn.Linear(in_features, nb_classes)
        elif model_type == 'mlp':
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, nb_classes)
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        if self.model_type == 'mlp':
            x = F.relu(x)
            x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    """Train the model for one epoch.
    
    Args:
        model: nn.Module, the model to train.
        device: str, either 'cpu' or 'cuda'.
        train_loader: torch.utils.data.DataLoader, the training data.
        optimizer: torch.optim.Optimizer, the optimizer to use.
        epoch: int, the current epoch.
        log_interval: int, how often to log.
    """
    model.train()
    total_correct = 0
    num_examples = 0
    tqdm.tqdm._instances.clear()  # handle plotting errors.
    for batch_idx, (data, target) in (pbar := tqdm.tqdm(enumerate(train_loader), total=len(train_loader))):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        correct = (output.argmax(dim=1) == target).sum().item()
        total_correct += correct
        num_examples += len(data)
        acc = total_correct / num_examples * 100
        msg = f'Train Epoch: {epoch}\tAccuracy: {acc:.3f}%\tLoss: {loss.item():.6f}'
        pbar.set_description(msg)
    pbar.close()


def evaluate(model, device, test_loader):
    """Evaluate the model on the test set.

    Args:
        model: nn.Module, the model to evaluate.
        device: str, either 'cpu' or 'cuda'.
        test_loader: torch.utils.data.DataLoader, the test data.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('', flush=True)
    acc_perc = 100. * correct / len(test_loader.dataset)
    print(f'Val set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc_perc:.3f}%)', flush=True)


def train_genders(
        datasets, genders,
        batch_size, test_batch_size,
        learning_rate, epochs,
        train_on_validation_set, model_type,
        *,
        state_dict=None,
        save_path_and_name=None):
    '''
    Train a model to predict genders based on provided datasets.

    Parameters:
    - datasets (tuple): Tuple of datasets (features, val_features, test_features) used for training, validation, and testing respectively.
    - genders (tuple): Tuple containing tensors for training, validation, and test gender labels.
    - batch_size (int): Size of batches used during training.
    - test_batch_size (int): Size of batches used during testing.
    - learning_rate (float): Learning rate used for optimization.
    - epochs (int): Number of training epochs.
    - train_on_validation_set (bool): If True, uses validation set also for training.
    - model_type (str): The type of the model ("mlp" or "linear").
    
    Keyword arguments:
    - state_dict (dict, optional): State dictionary to load into model before training.
    - save_path_and_name (str, optional): Path and filename where the model state_dict should be saved after training.
    
    Returns:
    - model: Trained PyTorch model.
    '''
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    features, val_features, test_features = datasets

    def to_cuda_tensor(df):
      return torch.Tensor(df.to_numpy()).type(torch.LongTensor).to(device)

    genders = tuple(map(to_cuda_tensor, genders))
    gender_train_tensor, gender_val_tensor, gender_test_tensor = genders

    if train_on_validation_set:
        x_train = torch.cat([features, val_features])
        y_train = torch.cat([gender_train_tensor, gender_val_tensor])
    else:
        x_train = features
        y_train = gender_train_tensor
    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(test_features, gender_test_tensor)
    train_loader = torch.utils.data.DataLoader(ds_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, **test_kwargs)

    in_features = features.shape[1]

    model = LogisticMLP(in_features=in_features, hidden_features=128, nb_classes=2, model_type=model_type).to(device)
    
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}", flush=True)
        evaluate(model, device, test_loader)
        train(model, device, train_loader, optimizer, epoch)
        # scheduler.step()
    
    print("Final evaluation on the test set:")
    evaluate(model, device, test_loader)

    if save_path_and_name is not None:
        torch.save(model.state_dict(), save_path_and_name)

    return model


def train_occupations(
        datasets, occupations,
        batch_size, val_batch_size,
        learning_rate, epochs,
        train_on_validation_set, model_type,
        *,
        state_dict=None,
        save_path_and_name=None):
    '''
    Train a model to predict occupations based on provided datasets.

    Parameters:
    - datasets (tuple): Tuple of datasets (features, val_features, test_features) used for training, validation, and testing.
    - occupations (tuple): Tuple containing labels for training, validation, and testing.
    - batch_size (int): Size of batches used during training.
    - val_batch_size (int): Size of batches used during validation.
    - learning_rate (float): Learning rate used for optimization.
    - epochs (int): Number of training epochs.
    - train_on_validation_set (bool): If True, uses the validation set also for training.
    - model_type (str): Type of the model ("mlp" or "linear").
    
    Keyword arguments:
    - state_dict (dict, optional): State dictionary to load into model before training.
    - save_path_and_name (str, optional): Path and filename where the model state_dict should be saved.
    
    Returns:
    - model: Trained PyTorch model.
    '''
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size}
    val_kwargs = {'batch_size': val_batch_size}
    
    if use_cuda:
        cuda_kwargs = {'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    features, val_features, test_features = datasets

    def to_cuda_tensor(arr):
        if type(arr) == np.ndarray:
            return torch.Tensor(arr).type(torch.LongTensor).to(device)
        else:
            return arr.type(torch.LongTensor).to(device)
    
    occupations = tuple(map(to_cuda_tensor, occupations))
    train_labels, val_labels, test_labels = occupations
    
    #add:
    features = torch.Tensor(features)
    val_features = torch.Tensor(val_features)
    test_features = torch.Tensor(test_features)

    if train_on_validation_set:
        x_train = torch.cat([features, val_features])
        y_train = torch.cat([train_labels, val_labels])
        val_labels = test_labels
        val_features = test_features
    else:
        x_train = features
        y_train = train_labels
    
    ds_train = TensorDataset(x_train, y_train)
    ds_val = TensorDataset(val_features, val_labels)
    ds_test = TensorDataset(test_features, test_labels)
    
    train_loader = torch.utils.data.DataLoader(ds_train, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(ds_val, **val_kwargs)
    test_loader = torch.utils.data.DataLoader(ds_test, **val_kwargs)

    in_features = features.shape[1]

    model = LogisticMLP(in_features=in_features, hidden_features=128, nb_classes=28, model_type=model_type).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}")
        evaluate(model, device, val_loader)
        train(model, device, train_loader, optimizer, epoch)

    print("Final evaluation on test set:")
    evaluate(model, device, test_loader)

    if save_path_and_name is not None:
        torch.save(model.state_dict(), save_path_and_name)

    return model


def predict_gender_from_occupation(occupation_dataset, genders):
    '''
    Predict gender from occupation using a co-occurrence matrix approach.

    Parameters:
    - occupation_dataset (tuple): Tuple containing labels for training, validation, and testing.
    - genders (tuple): Tuple containing gender labels for training, validation, and testing.
    
    Output:
    - Prints accuracy of prediction and plots a co-occurrence matrix.
    '''
    labels, val_labels, _ = occupation_dataset
    def trainable_labels(tensor):
        return tensor.long().cpu().numpy()    
    
    gender_train, gender_val, gender_test = genders
    co_occurrence_matrix = np.zeros((28, 2))
    y_occupations = np.concatenate([trainable_labels(labels), trainable_labels(val_labels)])
    y_gender = np.concatenate([gender_train.to_numpy(), gender_val.to_numpy()])
    y_occupations = y_occupations.flatten()
    y_gender = y_gender.flatten()
    np.add.at(co_occurrence_matrix, (y_occupations, y_gender), 1)
    predicted_gender = np.argmax(co_occurrence_matrix, axis=1)
    total_examples = np.sum(co_occurrence_matrix)
    correct_predictions = co_occurrence_matrix[np.arange(28), predicted_gender]
    accuracy = np.sum(correct_predictions) / total_examples
    print(f"Accuracy: {accuracy*100:.3f}%")
    plot_co_occurence(co_occurrence_matrix, per_class=True)