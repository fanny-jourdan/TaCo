import torch  
from tqdm.autonotebook import tqdm
from TaCo_baseline.tools.datasets_tools import load_dataset, create_splits
from TaCo_baseline.tools.datasets_tools import get_occupation_labels, load_embeddings
from TaCo_baseline.tools.model_utils import get_model


from TaCo_baseline.tools.train import train_genders
from TaCo_baseline.tools.train import train_occupations
from TaCo_baseline.tools.train import LogisticMLP
import numpy as np


from sklearn.linear_model import SGDClassifier, SGDRegressor, Perceptron, LogisticRegression

import random
import copy
import time

from sklearn.svm import LinearSVC
from TaCo_baseline.INLP.debias import get_debiasing_projection

from sklearn.metrics import accuracy_score

baseline = 'normal' 

#modeltype, nb_epochs = 'RoBERTa', 10
#modeltype, nb_epochs = 'DeBERTa', 3
modeltype, nb_epochs = 'DistilBERT', 3

basesavepath = "/data/fanny.jourdan/TaCo_baseline/"

if baseline == "normal":
  datafolder = "/datasets/shared_datasets/BIOS/"
  model_path = f"/datasets/shared_datasets/BIOS/models/{modeltype}_occBIOS_{nb_epochs}epochs_g1"  # predict occ

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dt_X, gender_names, occ_names = load_dataset(datafolder, baseline=baseline)
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



l_num_clfs = [10,20,50,100,200,300,500]
l_acc_occ, l_acc_gen = [], []

for num_clfs in l_num_clfs:
    idx = np.random.rand(x_train.shape[0]) < 1.
    P, rowspace_projections, Ws = get_projection_matrix(num_clfs, x_train[idx], y_train_gender[idx], x_dev, y_dev_gender, y_train, y_dev)

    np.save(basesavepath + f"INLP/proj/P_{num_clfs}_{modeltype}.npy", P)
    np.save(basesavepath + f"INLP/proj/rowspace_projections_{num_clfs}_{modeltype}.npy", rowspace_projections)
    np.save(basesavepath + f"INLP/proj/Ws_{num_clfs}_{modeltype}.npy", Ws)

    x_train_p, x_dev_p, x_test_p = (P.dot(x_train.T)).T, (P.dot(x_dev.T)).T, (P.dot(x_test.T)).T

    #nu information for the occupation
    save_path_occ = basesavepath + f'INLP/no_gender_pred/pred_occ{num_clfs}d_{modeltype}_b_{baseline}.pt'
    real_dataset = x_train_p, x_dev_p, x_test_p
    occupations = train_labels, val_labels, test_labels
    pocc_no_gender_model = train_occupations(real_dataset, occupations,
                                            batch_size=2048, val_batch_size=8192,
                                            learning_rate=5e-4, epochs=100,
                                            train_on_validation_set=False,
                                            model_type='mlp',
                                            save_path_and_name=save_path_occ)
    

    pred_occ = pocc_no_gender_model(x_test_p)
    _, predicted_classes = torch.max(pred_occ, 1)
    predicted_classes = predicted_classes.cpu().numpy()
    l_acc_occ.append(accuracy_score(predicted_classes, test_labels.cpu().numpy()))

    
    #nu information for the gender
    save_path_gen = basesavepath + f'INLP/no_gender_pred/pred_gender{num_clfs}d_{modeltype}_b_{baseline}.pt'
    def to_cuda_tensor(df):
        return torch.Tensor(df).type(torch.FloatTensor)#.to("cuda")
    real_dataset_tens = tuple(map(to_cuda_tensor, real_dataset))
    pg_no_gender_model = train_genders(real_dataset_tens, genders,
                                        batch_size=2048, test_batch_size=8192,
                                        learning_rate=5e-4, epochs=100,
                                        train_on_validation_set=True,
                                        model_type='mlp',
                                        save_path_and_name=save_path_gen)

    pred_gen = pg_no_gender_model(x_test_p)
    predicted_classes = torch.max(pred_gen, 1)
    predicted_classes = predicted_classes.cpu().numpy()
    l_acc_gen.append(accuracy_score(predicted_classes, gender_test))

l_results = {"l_acc_occ": l_acc_occ, "l_acc_gen": l_acc_gen, "l_num_clfs": l_num_clfs}
np.save(basesavepath + f'INLP/results/results_occ_gen_clfs_{modeltype}_b_{baseline}.npy', l_results)