import torch  
from datasets_tools import load_dataset, create_splits
from datasets_tools import get_occupation_labels, load_embeddings
from model_utils import get_model
from utils import batch_predict
from TaCo.TaCo.decomposition import decompose_choice
from TaCo.TaCo.concept_removal import sobol_importance_from_sample
from TaCo.TaCo.concept_removal import crop_concepts, build_gender_neutral_features
from train import train_genders, train_occupations
from train import LogisticMLP

import pickle
from sklearn.metrics import accuracy_score
import numpy as np

baseline = 'normal' 
#baseline = 'nogender'

#modeltype, nbepochs = 'RoBERTa', 10
#modeltype, nbepochs = 'DistilBERT', 3
modeltype, nbepochs = 'DeBERTa', 3

nb_reps = 5

datafolder = "/Users/fannyjourdan/Documents/doctorat/jupyterlab_OSIRIM/data"
if baseline == "normal":
  model_path = f"/Users/fannyjourdan/Documents/doctorat/jupyterlab_OSIRIM/models/BIOS_occupations_prediction/{modeltype}_occBIOS_{nb_epochs}epochs_g1"  # predict occ
elif baseline == 'nogender':
  model_path = f"/Users/fannyjourdan/Documents/doctorat/jupyterlab_OSIRIM/models/BIOS_occupations_prediction/{modeltype}_occBIOS_{nb_epochs}epochs_ng1"  # predict occ

else:
  print("Baseline not found")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
mlp_or_lin = "mlp"

##########################################IMPORT MODEL/DATA##########################################

dt_X, gender_names, occ_names = load_dataset(datafolder, baseline=baseline)
splits, genders = create_splits(dt_X)
model, tokenizer = get_model(model_path, model_type = modeltype)
dt_X_train, dt_X_val, dt_X_test = splits
gender_train, gender_val, gender_test = genders

datasets = dt_X_train, dt_X_val, dt_X_test
train_val_test_features, train_val_test_labels = load_embeddings(datasets,
                                                                 model=model,
                                                                 tokenizer=tokenizer,
                                                                 baseline=baseline,
                                                                 regenerate=False,
                                                                 model_type=modeltype,
                                                                 device=device)



train_features, val_features, test_features = train_val_test_features
train_labels, val_labels, test_labels = train_val_test_labels


train_labels, val_labels, test_labels = get_occupation_labels(dt_X_train, dt_X_val, dt_X_test, device)



##########################################DECOMPOSITION##########################################

features = torch.cat((train_features, val_features, test_features), dim=0)

#refill the tensors:
n_train, n_val, n_test = len(train_features), len(val_features), len(test_features)

method_name = "sSVD"
num_components = 18
decomposition_method = decompose_choice(method_name, num_components)


U, W = decomposition_method.decompose(features)

num_components = U.shape[1]
print("num_components:", num_components)

U_train, U_val, U_test = U.split((n_train, n_val, n_test), dim=0)


##########################################IMPORT SOBOL PART#########################################

global_importance_gender = pickle.load(open(f'global_importance/gi_gender_{modeltype}_{method_name}{num_components}_{mlp_or_lin}_baseline_{baseline}.pkl',"rb"))

global_importance_occ = pickle.load(open(f'global_importance/gi_occupation_{modeltype}_{method_name}{num_components}_{mlp_or_lin}_baseline_{baseline}.pkl',"rb"))


##########################################TRAIN WITH CONCEPT REMOVAL##########################################

angle = np.arctan(global_importance_occ / global_importance_gender) * 180 / np.pi

def to_cuda_tensor(arr):
    return torch.Tensor(arr).type(torch.FloatTensor).to(device)
    

l_gender_acc = []
l_occupation_acc = []

nb_epochs_training = 100

for nb_cpt_remov in range(1, num_components):
  W_no_gender, to_keep = crop_concepts(W.numpy(), angle, num_or_threshold=num_components-nb_cpt_remov)
  train_a_no_gender = build_gender_neutral_features(U_train.numpy(), W_no_gender, to_keep)
  val_a_no_gender = build_gender_neutral_features(U_val.numpy(), W_no_gender, to_keep)
  test_a_no_gender = build_gender_neutral_features(U_test.numpy(), W_no_gender, to_keep)
  train_a_no_gender.shape, val_a_no_gender.shape, test_a_no_gender.shape
    
  real_dataset = train_a_no_gender, val_a_no_gender, test_a_no_gender

  save_name = f'no_gender/pred_g_{method_name}{num_components}_cr{nb_cpt_remov}_{modeltype}_b_{baseline}.pt'
  real_dataset = tuple(map(to_cuda_tensor, real_dataset))

  save_path = f'no_gender/pred_occ_{method_name}{num_components}_cr{nb_cpt_remov}_{modeltype}_b_{baseline}.pt'
  occupations = train_labels, val_labels, test_labels

  test_a_no_gender = torch.from_numpy(test_a_no_gender).float().to(device)

  gender_acc = []
  occupation_acc = []
    
  for rep in range(nb_reps):
    #gender train
    pg_no_gender_model = train_genders(real_dataset, genders,
                                       batch_size=2048, test_batch_size=8192,
                                       learning_rate=5e-4, epochs=nb_epochs_training,
                                       train_on_validation_set=False,
                                       model_type=mlp_or_lin,
                                       save_path_and_name=save_name)
    
    pred_gen = pg_no_gender_model(test_a_no_gender)
    predicted_classes = torch.max(pred_gen, 1)
    predicted_classes = predicted_classes.cpu().numpy()
    
    gender_acc.append(accuracy_score(predicted_classes, gender_test))
       
    #occupation train
    pocc_no_gender_model = train_occupations(real_dataset, occupations,
                                            batch_size=2048, val_batch_size=8192,
                                            learning_rate=5e-4, epochs=nb_epochs_training,
                                            train_on_validation_set=False,
                                            model_type=mlp_or_lin,
                                            save_path_and_name=save_path)
       
    pred_occ = pocc_no_gender_model(test_a_no_gender)
    _, predicted_classes = torch.max(pred_occ, 1)
    predicted_classes = predicted_classes.cpu().numpy()

    occupation_acc.append(accuracy_score(predicted_classes, test_labels.cpu().numpy()))

  l_gender_acc.append(gender_acc)
  l_occupation_acc.append(occupation_acc)
    
pickle.dump(l_gender_acc, open(f'figures/list/l_gender_acc_{modeltype}_{method_name}{num_components}_{mlp_or_lin}_baseline_{baseline}_{nb_reps}reps.pkl',"wb"))
pickle.dump(l_occupation_acc, open(f'figures/list/l_occupation_acc_{modeltype}_{method_name}{num_components}_{mlp_or_lin}_baseline_{baseline}_{nb_reps}reps.pkl',"wb"))
