import sys
sys.path.append('/home/fanny.jourdan/dev/TaCo')


import torch  
from tools.datasets_tools import load_dataset, create_splits, load_embeddings
from tools.model_utils import get_model
from tools.train import train_genders, train_occupations

from TaCo import found_concepts, remove_concept_on_clstoken

import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

baseline = 'normal' 
#baseline = 'nogender'

#modeltype, nbepochs = 'RoBERTa', 10
#modeltype, nbepochs = 'DistilBERT', 3
#modeltype, nbepochs = 'DeBERTa', 3
modeltype, nbepochs = 't5', 2


basesavepath = "/data/fanny.jourdan/TaCo_baseline/"

if baseline == "normal":
  datafolder = "/datasets/shared_datasets/BIOS/"
  model_path = f"/datasets/shared_datasets/BIOS/models/{modeltype}_occBIOS_{nbepochs}epochs_g1"  # predict occ
elif baseline == 'nogender':
  datafolder = "/datasets/shared_datasets/BIOS_ng/"
  model_path = f"/datasets/shared_datasets/BIOS_ng/models/{modeltype}_occBIOS_{nbepochs}epochs_ng1"  # predict occ
else:
  print("Baseline not found")


nb_reps = 5
num_components = 20
method_decompose = "PCA"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
mlp_or_lin = "mlp"
nb_epochs_training = 100



##########################################IMPORT MODEL/DATA##########################################

# Load data
dt_X, gender_names, occ_names = load_dataset(datafolder)
splits, genders = create_splits(dt_X)
dt_X_train, dt_X_val, dt_X_test = splits
gender_train, gender_val, gender_test = genders
datasets = dt_X_train, dt_X_val, dt_X_test


# Load model
model, tokenizer = get_model(model_path, model_type = modeltype)

  
# Load CLS token embeddings
train_val_test_clstoken, train_val_test_labels = load_embeddings(datasets,
                                                                 baseline = baseline,
                                                                 model=model,
                                                                 tokenizer=tokenizer,
                                                                 path=basesavepath,
                                                                 regenerate=False,
                                                                 model_type=modeltype,
                                                                 nbatch=64,
                                                                 device=device)



train_clstoken, val_clstoken, test_clstoken = train_val_test_clstoken
train_labels, val_labels, test_labels = train_val_test_labels

# Load regression model to predict sensitive variable (here gender)
real_dataset = (train_clstoken, val_clstoken, test_clstoken)
save_name = basesavepath + f'gender_pred/{modeltype}_{mlp_or_lin}_baseline_{baseline}.pt'

state_dict = torch.load(save_name, map_location=torch.device(device))

pg_model = train_genders(real_dataset, genders,
                          batch_size=2048, test_batch_size=8192,
                          learning_rate=1e-3, epochs=0,
                          train_on_validation_set=True,
                          model_type=mlp_or_lin,
                          save_path_and_name=save_name,
                          state_dict=state_dict)


##########################################DECOMPOSITION##########################################

U_train, U_val, U_test, W, angle = found_concepts(train_clstoken, val_clstoken, test_clstoken,
                                                  model, pg_model, device = device,
                                                  method_decompose = method_decompose, num_components = num_components, 
                                                  sobol_nb_design = 50, sobol_sampled = 10_000)


print("found concepts done")


def to_cuda_tensor(arr):
    return torch.Tensor(arr).type(torch.FloatTensor).to(device)
    

l_gender_acc, l_occupation_acc = [], []
l_gender_acc_lin, l_occupation_acc_lin = [], []


for nb_cpt_remov in range(1, num_components):

  train_clstoken_no_gender, val_clstoken_no_gender, test_clstoken_no_gender = remove_concept_on_clstoken(U_train, U_val, U_test, W, angle, nb_cpt_remov, num_components) 
  

  real_dataset = train_clstoken_no_gender, val_clstoken_no_gender, test_clstoken_no_gender

  real_dataset = tuple(map(to_cuda_tensor, real_dataset))
  occupations = train_labels, val_labels, test_labels

  test_clstoken_no_gender = torch.from_numpy(test_clstoken_no_gender).float().to(device)

  gender_acc, occupation_acc = [], []
  gender_acc_lin, occupation_acc_lin = [], []
    
  for rep in range(nb_reps):
    #nu information for the gender
    pg_no_gender_model = train_genders(real_dataset, genders,
                                       batch_size=2048, test_batch_size=8192,
                                       learning_rate=5e-4, epochs=nb_epochs_training,
                                       train_on_validation_set=False,
                                       model_type=mlp_or_lin)
    
    pred_gen = pg_no_gender_model(test_clstoken_no_gender)
    _, predicted_classes = torch.max(pred_gen, 1)
    predicted_classes = predicted_classes.cpu().numpy()
    
    gender_acc.append(accuracy_score(predicted_classes, gender_test))

    #linear nu information for the gender
    classifier_g = LogisticRegression(max_iter=1000)
    classifier_g.fit(train_clstoken_no_gender, gender_train)
    gender_acc_lin.append(classifier_g.score(test_clstoken_no_gender.cpu(), gender_test))
                        
    #nu information for the occupation
    pocc_no_gender_model = train_occupations(real_dataset, occupations,
                                            batch_size=2048, val_batch_size=8192,
                                            learning_rate=5e-4, epochs=nb_epochs_training,
                                            train_on_validation_set=False,
                                            model_type=mlp_or_lin)
       
    pred_occ = pocc_no_gender_model(test_clstoken_no_gender)
    _, predicted_classes = torch.max(pred_occ, 1)
    predicted_classes = predicted_classes.cpu().numpy()

    occupation_acc.append(accuracy_score(predicted_classes, test_labels.cpu().numpy()))

    #linear nu information for the occupation
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_clstoken_no_gender, train_labels.cpu())
    occupation_acc_lin.append(classifier.score(test_clstoken_no_gender.cpu(), test_labels.cpu())) 

  l_gender_acc.append(gender_acc)
  l_occupation_acc.append(occupation_acc)
  l_gender_acc_lin.append(gender_acc_lin)
  l_occupation_acc_lin.append(occupation_acc_lin) 
    
pickle.dump(l_gender_acc, open(basesavepath + f'TaCo/list_results/l_gender_acc_{modeltype}_{method_decompose}{num_components}_{mlp_or_lin}_baseline_{baseline}_{nb_reps}reps.pkl',"wb"))
pickle.dump(l_occupation_acc, open(basesavepath + f'TaCo/list_results/l_occupation_acc_{modeltype}_{method_decompose}{num_components}_{mlp_or_lin}_baseline_{baseline}_{nb_reps}reps.pkl',"wb"))
pickle.dump(l_gender_acc_lin, open(basesavepath + f'TaCo/list_results/l_LIN_gender_acc_{modeltype}_{method_decompose}{num_components}_{mlp_or_lin}_baseline_{baseline}_{nb_reps}reps.pkl',"wb"))
pickle.dump(l_occupation_acc_lin, open(basesavepath + f'TaCo/list_results/l_LIN_occupation_acc_{modeltype}_{method_decompose}{num_components}_{mlp_or_lin}_baseline_{baseline}_{nb_reps}reps.pkl',"wb"))
