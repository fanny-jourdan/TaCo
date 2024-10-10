# Please note the code is inspired from the following repository: 
# https://github.com/shauli-ravfogel/nullspace_projection/notebooks/biasbios_bert.ipynb


import sys
sys.path.append('/home/fanny.jourdan/dev/TaCo')

import torch  
from tqdm.autonotebook import tqdm
from tools.datasets_tools import load_dataset, create_splits, get_occupation_labels, load_embeddings
from tools.model_utils import get_model

from tools.train import train_genders, train_occupations, LogisticMLP
import numpy as np

from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron, LogisticRegression

import random
import copy
import time

from sklearn.svm import LinearSVC
from baselines.INLP.debias import get_debiasing_projection

from sklearn.metrics import accuracy_score

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


train_labels, val_labels, test_labels = get_occupation_labels(dt_X_train, dt_X_val, dt_X_test, device)


x_train = train_features.cpu().numpy()
y_train_gender = gender_train.to_numpy()
y_train = train_labels.cpu().numpy() 

x_dev = val_features.cpu().numpy()
y_dev_gender = gender_val.to_numpy()
y_dev = val_labels.cpu().numpy()

x_test = test_features.cpu().numpy()
y_test_gender = gender_test.to_numpy()
y_test = test_labels.cpu().numpy()


def get_projection_matrix(num_clfs, X_train, Y_train_gender, X_dev, Y_dev_gender, Y_train_task, Y_dev_task):

    is_autoregressive = True
    min_acc = 0.
    n = num_clfs
    #random_subset = 1.0
    start = time.time()
    TYPE= "svm"
    dim = X_train.shape[1]
    
    #x_train_gender = x_train.copy()
    #x_dev_gender = x_dev.copy()
        
    
    if TYPE == "sgd":
        gender_clf = SGDClassifier
        params = {'loss': 'hinge', 'penalty': 'l2', 'fit_intercept': False, 'class_weight': None, 'n_jobs': 32}
    else:
        gender_clf = LinearSVC
        params = {'penalty': 'l2', 'C': 0.01, 'fit_intercept': True, 'class_weight': None, "dual": False}
        
    P,rowspace_projections, Ws = get_debiasing_projection(gender_clf, params, n, dim, is_autoregressive, min_acc,
                                              X_train, Y_train_gender, X_dev, Y_dev_gender,
                                       Y_train_main=Y_train_task, Y_dev_main=Y_dev_task, by_class = True)
    print("time: {}".format(time.time() - start))
    return P,rowspace_projections, Ws



l_num_clfs = [200,300,400,500,550,600,650,700,750]

nb_reps = 5 
l_acc_occ, l_acc_gen, l_num_clfs_aux = [], [], []
l_acc_occ_linear, l_acc_gen_linear = [], []

for num_clfs in l_num_clfs:
    l_num_clfs_aux.append(num_clfs)
    idx = np.random.rand(x_train.shape[0]) < 1.
    P, rowspace_projections, Ws = get_projection_matrix(num_clfs, x_train[idx], y_train_gender[idx], x_dev, y_dev_gender, y_train, y_dev)

    np.save(basesavepath + f"INLP/proj/P_{num_clfs}_{modeltype}.npy", P)
    np.save(basesavepath + f"INLP/proj/rowspace_projections_{num_clfs}_{modeltype}.npy", rowspace_projections)
    np.save(basesavepath + f"INLP/proj/Ws_{num_clfs}_{modeltype}.npy", Ws)

    #P = np.load(basesavepath + f"INLP/proj/P_{num_clfs}_{modeltype}.npy", allow_pickle=True)

    x_train_p, x_dev_p, x_test_p = (P.dot(x_train.T)).T, (P.dot(x_dev.T)).T, (P.dot(x_test.T)).T

    real_dataset = x_train_p, x_dev_p, x_test_p
    x_test_p_tensor = torch.Tensor(x_test_p).type(torch.FloatTensor).to(device)

    def to_cuda_tensor(df):
            return torch.Tensor(df).type(torch.FloatTensor)#.to("cuda")
    real_dataset_tens = tuple(map(to_cuda_tensor, real_dataset))

    occupations = train_labels, val_labels, test_labels

    acc_occ, acc_gen, acc_occ_linear, acc_gen_linear = [], [], [], []

    for rep in range(nb_reps):
        #linear nu information for the occupation
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(x_train_p, y_train)
        acc_occ_linear.append(classifier.score(x_test_p, y_test))

        #nu information for the occupation
        pocc_no_occ_model = train_occupations(real_dataset, occupations,
                                                batch_size=2048, val_batch_size=8192,
                                                learning_rate=5e-4, epochs=100,
                                                train_on_validation_set=False,
                                                model_type='mlp')
    

        pred_occ = pocc_no_occ_model(x_test_p_tensor)
        _, predicted_classes = torch.max(pred_occ, 1)
        predicted_classes = predicted_classes.cpu().numpy()
        acc_occ.append(accuracy_score(predicted_classes, test_labels.cpu().numpy()))

        #linear nu information for the gender
        classifier_g = LogisticRegression(max_iter=1000)
        classifier_g.fit(x_train_p, y_train_gender)
        acc_gen_linear.append(classifier_g.score(x_test_p, y_test_gender))

        #nu information for the gender
        pg_no_gender_model = train_genders(real_dataset_tens, genders,
                                            batch_size=2048, test_batch_size=8192,
                                            learning_rate=5e-4, epochs=100,
                                            train_on_validation_set=True,
                                            model_type='mlp')

        pred_gen = pg_no_gender_model(x_test_p_tensor)
        _, predicted_classes = torch.max(pred_gen, 1)
        predicted_classes = predicted_classes.cpu().numpy()
        acc_gen.append(accuracy_score(predicted_classes, gender_test))

    l_acc_occ_linear.append(acc_occ_linear)    
    l_acc_gen_linear.append(acc_gen_linear)
    l_acc_occ.append(acc_occ)   
    l_acc_gen.append(acc_gen)
    l_results = {"l_acc_occ": l_acc_occ, "l_acc_gen": l_acc_gen, "l_num_clfs": l_num_clfs_aux,
                 "l_acc_occ_linear": l_acc_occ_linear, "l_acc_gen_linear": l_acc_gen_linear}
    np.save(basesavepath + f'INLP/results/results_occ_gen_clfs_{modeltype}_b_{baseline}_{nb_reps}reps.npy', l_results)


