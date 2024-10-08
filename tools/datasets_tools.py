import numpy as np
import pickle
import pandas as pd
import torch
from fast_ml.model_development import train_valid_test_split

from tools.utils import batch_predict


def get_occupation_names():
  occ_names = ['surgeon', 'pastor', 'photographer', 'professor', 'chiropractor', 'software_engineer', 'teacher', 'poet', 'dj', 'rapper', 'paralegal', 'physician', 
              'journalist', 'architect', 'attorney', 'yoga_teacher', 'nurse', 'painter', 'model', 'composer', 'personal_trainer','filmmaker', 'comedian', 'accountant', 
              'interior_designer', 'dentist', 'psychologist', 'dietitian']
  return occ_names


def get_gender_names():
  gender_names = ['F', 'M']
  return gender_names


def load_dataset(datafolder):
  """Load the dataset.
  
  Args:
    datafolder: path to the folder containing the dataset
  
  Returns:
    a tuple (dt_X, gender_names, occ_names)
      where dt_X is a dataframe of strings,
      gender_names and occ_names are list of strings
  """  

  X = pickle.load(open(f"{datafolder}/data.pkl",'rb'))

  gender = pickle.load(open(f"{datafolder}/gender.pkl",'rb'))
  labels = pickle.load(open(f"{datafolder}/labels.pkl",'rb'))

  gender_names = get_gender_names()
  occ_names = get_occupation_names()

  dico_gen = {gender_name: i for i, gender_name in enumerate(gender_names)}
  dico_lab = {occ_name: i for i, occ_name in enumerate(occ_names)}

  dt_X = pd.DataFrame(X, columns=['sentence'])
  dt_X["gender"] = [dico_gen[gen] for gen in gender]
  dt_X["label"] = [dico_lab[lab] for lab in labels] 

  return dt_X, gender_names, occ_names


def create_splits(dt_X):
  """Create train, validation and test splits.
  
  Args:
    dt_X: a dataframe of strings
  
  Returns:
    a tuple (splits, genders) with splits a tuple of dataframes,
    and genders of gender labels.
  """
  splits = train_valid_test_split(dt_X, target = 'label',  method='random',
                                  train_size=0.7, valid_size=0.1, test_size=0.2,
                                  random_state=0)

  X_train, y_train, X_valid, y_valid, X_test, y_test = splits

  gender_train = X_train["gender"]
  gender_val = X_valid["gender"]
  gender_test = X_test["gender"]

  dt_X_train = pd.DataFrame(X_train["sentence"], columns= ['sentence'])
  dt_X_train["label"] = y_train
  dt_X_train = np.array(dt_X_train) 

  dt_X_val = pd.DataFrame(X_valid["sentence"], columns= ['sentence'])
  dt_X_val["label"] = y_valid
  dt_X_val = np.array(dt_X_val) 

  dt_X_test = pd.DataFrame(X_test["sentence"], columns= ['sentence'])
  dt_X_test["label"] = y_test
  dt_X_test = np.array(dt_X_test)    

  splits = dt_X_train, dt_X_val, dt_X_test
  genders = gender_train, gender_val, gender_test
  return splits, genders


def get_occupation_labels(dt_X_train, dt_X_val, dt_X_test, device = 'cuda'):
  labels = torch.Tensor(dt_X_train[:,1].astype(int)).to(device)
  val_labels = torch.Tensor(dt_X_val[:,1].astype(int)).to(device)
  test_labels = torch.Tensor(dt_X_test[:,1].astype(int)).to(device)
  return labels, val_labels, test_labels


def load_in_chunks(file_name, device, chunk_size=1000):
    # CPU
    tensor = torch.load(file_name, map_location=torch.device("cpu"))

    # chunks for loading on GPU
    num_chunks = (tensor.shape[0] + chunk_size - 1) // chunk_size
    tensor_cuda = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, tensor.shape[0])
        chunk = tensor[start_idx:end_idx].to(device)
        tensor_cuda.append(chunk)

    # Concatenate chunks
    tensor_cuda = torch.cat(tensor_cuda, dim=0)
    return tensor_cuda


def load_embeddings(dataframes, baseline, *,
                    model=None,
                    tokenizer=None,
                    path=None,
                    regenerate=False,
                    model_type="RoBERTa",
                    nbatch=128,
                    device='cuda'):
  """Load the embeddings on the disk.
  
  Args:
    dataframes: A tuple of dataframes (train, val, test).
    model: A model that has a features attribute (default: None).
    regenerate: If True, regenerate the embeddings (default: False).
    model_type = Type of model loaded "RoBERTa" or "DeBERTa" (default: "RoBERTa").
    cuda: If True, load embeddings on the GPU (default: True).
    nbatch: Size of batch for the prediction, int (default: 128).
    baseline: str, 'normal' or 'nogender'.
  """
  if baseline == 'normal':
        train_features_name = path + f'features/train_{model_type}embeddings.pt'  
        val_features_name = path + f'features/val_{model_type}embeddings.pt'  
        test_features_name = path + f'features/test_{model_type}embeddings.pt'
  elif baseline == 'nogender':
        train_features_name = path + f'features/train_{model_type}embeddings_negi.pt'  
        val_features_name = path + f'features/val_{model_type}embeddings_negi.pt'  
        test_features_name = path + f'features/test_{model_type}embeddings_negi.pt'
  else:
    assert False, "baseline must be 'normal' or 'nogender'"

  dt_X_train, dt_X_val, dt_X_test = dataframes

  if regenerate:
    train_features, _ = batch_predict(model.features, tokenizer, dt_X_train, nbatch, device)
    torch.save(train_features, train_features_name)
    val_features, _ = batch_predict(model.features, tokenizer, dt_X_val, nbatch, device)
    torch.save(val_features, val_features_name)
    test_features, _ = batch_predict(model.features, tokenizer, dt_X_test, nbatch, device)
    torch.save(test_features, test_features_name)
  else:
    train_features = load_in_chunks(train_features_name, device)
    val_features = load_in_chunks(val_features_name, device)
    test_features = load_in_chunks(test_features_name, device)

  train_val_test_features = train_features, val_features, test_features
  train_val_test_labels = get_occupation_labels(dt_X_train, dt_X_val, dt_X_test, device)

  return train_val_test_features, train_val_test_labels
