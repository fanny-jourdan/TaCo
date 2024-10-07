
import torch
import numpy as np

from decomposition import decompose_choice
from concept_removal import sobol_importance_from_sample
from concept_removal import crop_concepts, build_gender_neutral_features


def found_concepts(train_clstoken, val_clstoken, test_clstoken,
                   model, pg_model, device = "cuda",
                   method_decompose = "SVD", num_components = 20, 
                   sobol_nb_design = 50, sobol_sampled = 10_000):
    """
    Args:
        train_clstoken: torch.Tensor
        val_clstoken: torch.Tensor
        test_clstoken: torch.Tensor

        model: torch.nn.Module (all the model)
        pg_model: torch.nn.Module (just the classification between CLS token and sensitive attribute)

        PART1: DECOMPOSITION
        method_decompose: one of ("sSVD", "PCA", "ICA")
        num_components: int

        PART2: SOBOL
        sobol_nb_design: int
        sobol_sampled: int

    """
    
    features = torch.cat((train_clstoken, val_clstoken, test_clstoken), dim=0)
    n_train, n_val, n_test = len(train_clstoken), len(val_clstoken), len(test_clstoken)

    decomposition_method = decompose_choice(method_decompose, num_components)

    U, W = decomposition_method.decompose(features)

    U_train, U_val, U_test = U.split((n_train, n_val, n_test), dim=0)

    pg_model.eval() 
    global_importance_gender, _ = sobol_importance_from_sample(train_clstoken, U_train.numpy(),
                                                                                W.numpy(), pg_model,
                                                                                sampled=sobol_sampled,
                                                                                num_components=num_components,
                                                                                sobol_nb_design=sobol_nb_design, 
                                                                                device=device)
    model.eval()
    model_occ = lambda x: model.end_model(x)
    global_importance_occ, _ = sobol_importance_from_sample(train_clstoken, U_train.numpy(),
                                                            W.numpy(), model_occ,
                                                            sampled=sobol_sampled,
                                                            num_components=num_components,
                                                            sobol_nb_design=sobol_nb_design,
                                                            device=device)
    
    angle = np.arctan(global_importance_occ / global_importance_gender) * 180 / np.pi
    
    return U_train, U_val, U_test, W, angle



def remove_concept_on_clstoken(U_train, U_val, U_test, W, angle, nb_cpt_remov, num_components = 20):
    """
    Args:
        PART3: CONCEPT REMOVAL
        U_train: torch.Tensor
        U_val: torch.Tensor
        U_test: torch.Tensor
        W: torch.Tensor
        angle: np.array
        nb_cpt_remov: int
        num_components: int
    """ 
    if nb_cpt_remov >= num_components:
        raise ValueError("nb_cpt_remov should be less than num_components")
    
    W_no_gender, to_keep = crop_concepts(W.numpy(), angle, num_or_threshold=num_components-nb_cpt_remov)

    train_clstoken_no_gender = build_gender_neutral_features(U_train.numpy(), W_no_gender, to_keep)
    val_clstoken_no_gender = build_gender_neutral_features(U_val.numpy(), W_no_gender, to_keep)
    test_clstoken_no_gender = build_gender_neutral_features(U_test.numpy(), W_no_gender, to_keep)

    return train_clstoken_no_gender, val_clstoken_no_gender, test_clstoken_no_gender
