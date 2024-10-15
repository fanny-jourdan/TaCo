import sys
sys.path.append('/home/fanny.jourdan/dev/TaCo')

import torch  
from tools.datasets_tools import load_dataset, create_splits, get_occupation_labels, load_embeddings
from tools.model_utils import get_model

from tools.train import train_genders, train_occupations
import numpy as np



from baselines.RLACE.rlace import solve_adv_game

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

baseline = 'normal' 

#modeltype, nb_epochs = 'RoBERTa', 10
modeltype, nb_epochs = 'DeBERTa', 3
#modeltype, nb_epochs = 'DistilBERT', 3

basesavepath = "/data/fanny.jourdan/TaCo_baseline/"

if baseline == "normal":
  datafolder = "/datasets/shared_datasets/BIOS/"
  model_path = f"/datasets/shared_datasets/BIOS/models/{modeltype}_occBIOS_{nb_epochs}epochs_g1"  # predict occ

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dt_X, gender_names, occ_names = load_dataset(datafolder)
splits, genders = create_splits(dt_X)
model, tokenizer = get_model(model_path, model_type = modeltype)
dt_X_train, dt_X_val, dt_X_test = splits
gender_train, gender_val, gender_test = genders


datasets = dt_X_train, dt_X_val, dt_X_test
train_val_test_features, train_val_test_labels = load_embeddings(datasets,
                                                                 model=model,
                                                                 tokenizer=tokenizer,
                                                                 path=basesavepath,
                                                                 baseline=baseline,
                                                                 regenerate=False,
                                                                 model_type=modeltype,
                                                                 device=device)
train_features, val_features, test_features = train_val_test_features
train_labels, val_labels, test_labels = train_val_test_labels



x_train = train_features.cpu().numpy()
y_train_gender = gender_train.to_numpy()
y_train = train_labels.cpu().numpy() 

x_dev = val_features.cpu().numpy()
y_dev_gender = gender_val.to_numpy()
y_dev = val_labels.cpu().numpy()

x_test = test_features.cpu().numpy()
y_test_gender = gender_test.to_numpy()
y_test = test_labels.cpu().numpy()


#l_acc_occ, l_acc_gen, l_epsilon_aux = [], [], []
l_acc_occ, l_acc_gen, l_rank_aux = [], [], []
l_acc_occ_lin, l_acc_gen_lin = [], []


num_iters = 50000
#num_iters = 5000


#rank=1
l_rank = [1,5,10,20,50,100,200,500]
nb_reps = 5


optimizer_class = torch.optim.SGD
optimizer_params_P = {"lr": 0.003, "weight_decay": 1e-4}
optimizer_params_predictor = {"lr": 0.003,"weight_decay": 1e-4}

epsilon = 0.001 # stop 0.1% from majority acc
#l_epsilon = [0.0001, 0.001, 0.01, 0.1]

#batch_size = 256
#batch_size = 1024
#batch_size = 4096
batch_size = 8192


#for epsilon in l_epsilon:
for rank in l_rank:
    #l_epsilon_aux.append(epsilon)
    l_rank_aux.append(rank)
    dx = np.random.rand(x_train.shape[0]) < 1.
    output = solve_adv_game(x_train, y_train_gender, x_dev, y_dev_gender, rank=rank, device="cpu", out_iters=num_iters,
                       optimizer_class=optimizer_class, optimizer_params_P=optimizer_params_P,
                       optimizer_params_predictor=optimizer_params_predictor, epsilon=epsilon, batch_size=batch_size)
    


    np.save(basesavepath + f"RLACE/proj/output_{rank}rank_{num_iters}iters_epsilon{epsilon}_{modeltype}.npy", output)
    #output = np.load(basesavepath + f"RLACE/proj/output_{rank}rank_{num_iters}iters_epsilon{epsilon}_{modeltype}.npy", allow_pickle=True).item()
    
    P = output["P"]
    x_train_p, x_dev_p, x_test_p = x_train@P, x_dev@P, x_test@P

    acc_occ, acc_gen, acc_occ_lin, acc_gen_lin = [], [], [], []
    for rep in range(nb_reps):
       #nu information for the occupation
       save_path_occ = basesavepath + f'RLACE/no_gender_pred/pred_{rank}rank_occ{num_iters}iters_epsi{epsilon}_{modeltype}_b_{baseline}.pt'
       
       real_dataset = x_train_p, x_dev_p, x_test_p
       x_test_p_tensor = torch.Tensor(x_test_p).type(torch.FloatTensor).to(device)

       occupations = train_labels, val_labels, test_labels
       pocc_no_gender_model = train_occupations(real_dataset, occupations,
                                               batch_size=2048, val_batch_size=8192,
                                               learning_rate=5e-4, epochs=100,
                                               train_on_validation_set=False,
                                               model_type='mlp',
                                               save_path_and_name=save_path_occ)
    

       pred_occ = pocc_no_gender_model(x_test_p_tensor)
       _, predicted_classes = torch.max(pred_occ, 1)
       predicted_classes = predicted_classes.cpu().numpy()
       acc_occ.append(accuracy_score(predicted_classes, test_labels.cpu().numpy()))

       #linear nu information for the occupation
       classifier = LogisticRegression(max_iter=1000)
       classifier.fit(x_train_p, y_train)
       acc_occ_lin.append(classifier.score(x_test_p, y_test))

       #nu information for the gender
       save_path_gen = basesavepath + f'RLACE/no_gender_pred/pred_{rank}rank_gender{num_iters}iters_epsi{epsilon}_{modeltype}_b_{baseline}.pt'

       def to_cuda_tensor(df):
           return torch.Tensor(df).type(torch.FloatTensor)#.to("cuda")
       real_dataset_tens = tuple(map(to_cuda_tensor, real_dataset))
       pg_no_gender_model = train_genders(real_dataset_tens, genders,
                                           batch_size=2048, test_batch_size=8192,
                                           learning_rate=5e-4, epochs=100,
                                           train_on_validation_set=True,
                                           model_type='mlp',
                                           save_path_and_name=save_path_gen)

       pred_gen = pg_no_gender_model(x_test_p_tensor)
       _, predicted_classes = torch.max(pred_gen, 1)
       predicted_classes = predicted_classes.cpu().numpy()
       acc_gen.append(accuracy_score(predicted_classes, gender_test))

       #linear nu information for the gender
       classifier_g = LogisticRegression(max_iter=1000)
       classifier_g.fit(x_train_p, y_train_gender)
       acc_gen_lin.append(classifier_g.score(x_test_p, y_test_gender))
    
    l_acc_gen.append(acc_gen)
    l_acc_occ.append(acc_occ)
    l_acc_occ_lin.append(acc_occ_lin)
    l_acc_gen_lin.append(acc_gen_lin)
    l_results = {"l_acc_occ": l_acc_occ, "l_acc_gen": l_acc_gen, "rank": l_rank_aux,
                 "l_acc_occ_linear": l_acc_occ_lin, "l_acc_gen_linear": l_acc_gen_lin}
    
    np.save(basesavepath + f'RLACE/results/results_{num_iters}iters_{epsilon}epsilon_occ_gen_rank_{modeltype}_b_{baseline}_{nb_reps}reps.npy', l_results)



#l_results = {"l_acc_occ": l_acc_occ, "l_acc_gen": l_acc_gen, "epsilon": l_epsilon}
#np.save(basesavepath + f'RLACE/results/results__{rank}rank_{num_iters}iters_occ_gen_epsilon_{modeltype}_b_{baseline}.npy', l_results)