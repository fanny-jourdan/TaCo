import sys
sys.path.append('/home/fanny.jourdan/dev/TaCo')


import torch  
from tools.datasets_tools import load_dataset, create_splits
from tools.datasets_tools import load_embeddings
from tools.model_utils import get_model
from tools.train import train_genders, train_occupations

from sklearn.linear_model import LogisticRegression
from baselines.LEACE.leace import LeaceEraser
import pandas as pd 
import numpy as np

from sklearn.metrics import accuracy_score

baseline = 'normal' 

#modeltype, nb_epochs = 'RoBERTa', 10
#modeltype, nb_epochs = 'DeBERTa', 3
#modeltype, nb_epochs = 'DistilBERT', 3
modeltype, nb_epochs = 't5', 2


basesavepath = "/data/fanny.jourdan/TaCo_baseline/"

if baseline == "normal":
  datafolder = "/datasets/shared_datasets/BIOS/"
  model_path = f"/datasets/shared_datasets/BIOS/models/{modeltype}_occBIOS_{nb_epochs}epochs_g1"  # predict occ

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dt_X, gender_names, occ_names = load_dataset(datafolder)
splits, genders = create_splits(dt_X)
model, tokenizer = get_model(model_path, model_type = modeltype)
dt_X_train, dt_X_val, dt_X_test = splits
y_train_gender_t, y_dev_gender_t, y_test_gender_t = genders


datasets = dt_X_train, dt_X_val, dt_X_test
train_val_test_features, train_val_test_labels = load_embeddings(datasets,
                                                                 model=model,
                                                                 tokenizer=tokenizer,
                                                                 path=basesavepath,
                                                                 baseline=baseline,
                                                                 regenerate=False,
                                                                 model_type=modeltype,
                                                                 device=device)

x_train_t, x_dev_t, x_test_t = train_val_test_features
y_train_t, y_dev_t, y_test_t = train_val_test_labels



x_train = x_train_t.cpu().numpy()
y_train_gender = y_train_gender_t.to_numpy()
y_train = y_train_t.cpu().numpy() 

x_dev = x_dev_t.cpu().numpy()
y_dev_gender = y_dev_gender_t.to_numpy()
y_dev = y_dev_t.cpu().numpy()

x_test = x_test_t.cpu().numpy()
y_test_gender = y_test_gender_t.to_numpy()
y_test = y_test_t.cpu().numpy()



y_train_gender_t = torch.tensor(y_train_gender_t.values) if isinstance(y_train_gender_t, pd.Series) else y_train_gender_t
y_dev_gender_t = torch.tensor(y_dev_gender_t.values) if isinstance(y_dev_gender_t, pd.Series) else y_dev_gender_t


# Combine training and development sets if needed
x_leace = torch.cat([x_train_t.cpu(), x_dev_t.cpu()], dim=0)
y_leace_gender = torch.cat([y_train_gender_t.cpu(), y_dev_gender_t.cpu()], dim=0)

# Fit the LEACE eraser to your data
eraser = LeaceEraser.fit(x_leace, y_leace_gender)

# Erase gender from the embeddings
x_train_erased = eraser(x_train_t.cpu())
x_dev_erased = eraser(x_dev_t.cpu())
x_test_erased = eraser(x_test_t.cpu())

real_dataset = x_train_erased, x_dev_erased, x_test_erased
occupations = y_train_t, y_dev_t, y_test_t
def to_cuda_tensor(df):
   return torch.Tensor(df).type(torch.FloatTensor)#.to("cuda")
real_dataset_tens = tuple(map(to_cuda_tensor, real_dataset))

x_test_erased_tensor = torch.Tensor(x_test_erased).type(torch.FloatTensor).to(device)

l_acc_occ, l_acc_gender, l_acc_occ_linear, l_acc_gender_linear = [], [], [], []
nb_reps = 5

for rep in range(nb_reps):
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train_erased.numpy(), y_train)
    l_acc_occ_linear.append(classifier.score(x_test_erased.numpy(), y_test))
    
    classifier_g = LogisticRegression(max_iter=1000)
    classifier_g.fit(x_train_erased.numpy(), y_train_gender)
    l_acc_gender_linear.append(classifier_g.score(x_test_erased.numpy(), y_test_gender))
    
    pocc_no_occ_model = train_occupations(real_dataset, occupations,
                                             batch_size=2048, val_batch_size=8192,
                                             learning_rate=5e-4, epochs=100,
                                             train_on_validation_set=False,
                                             model_type='mlp')
    
    
    pred_occ = pocc_no_occ_model(x_test_erased_tensor)
    _, predicted_classes = torch.max(pred_occ, 1)
    predicted_classes = predicted_classes.cpu().numpy()
    l_acc_occ.append(accuracy_score(predicted_classes, y_test))
    
    pg_no_gender_model = train_genders(real_dataset_tens, genders,
                                       batch_size=2048, test_batch_size=8192,
                                       learning_rate=5e-4, epochs=100,
                                       train_on_validation_set=False,
                                       model_type='mlp')
    

    pred_gen = pg_no_gender_model(x_test_erased_tensor)
    _, predicted_classes = torch.max(pred_gen, 1)
    predicted_classes = predicted_classes.cpu().numpy()
    l_acc_gender.append(accuracy_score(predicted_classes, y_test_gender))

l_results = {"l_acc_occ": l_acc_occ, "l_acc_gen": l_acc_gender, 
             "l_acc_occ_linear": l_acc_occ_linear, "l_acc_gen_linear": l_acc_gender_linear}

np.save(basesavepath + f'LEACE/results/results_occ_gen_{modeltype}_b_{baseline}_{nb_reps}reps.npy', l_results)
   