import torch
import numpy as np

from TaCo.sobol import ScipySobolSequence, JansenEstimator
from math import ceil
import tqdm as tqdm
from sklearn.decomposition import NMF


def concept_perturbation(a, u, masks, W, model, device="cuda"):
  """Return output on perturbed concepts.

  Args:
    a: torch.Tensor of shape (num_features,)
    u: torch.Tensor of shape (num_components,)
    masks: torch.Tensor of shape (num_masks, num_components)
    W: torch.Tensor of shape (num_components, num_features)

  u_masked : (num_masks, num_components)
  a_sparse : (num_masks, num_features)
  y : (num_masks,)
  """
  delta = a - u @ W  # reconstruction error (residuals).

  masks = 1. - masks  # from [0, 1) to (0, 1] for less OOD
  u_masked = u[None, :] * masks  # u with masked concepts.
  a_sparse = u_masked @ W  # a with masked concepts.
  a_sparse = a_sparse + delta[None, :]  # re-inject residuals.

  errors = np.linalg.norm(a - a_sparse, axis=-1)
  rel_errors = errors / np.linalg.norm(a, axis=-1)

  with torch.no_grad():
    a_sparse = torch.Tensor(a_sparse).to(device)
    y = model(a_sparse)
    y = y.cpu().numpy()

    y = y.max(axis=1) - y[np.arange(len(y)), y.argsort(axis=1)[:,-2]] #difference between the maximum logit and the second maximum logit
    #y = y[:,0] - y[:,1]  # Difference of logits on MulticlassCrossEntropy is equivalent to BCE logits.
    # TODO: report accuracy drop instead of logits drop? Less sensitive to predictions.
  
  return y, rel_errors


def sobol_importance(A, U, W, model, sobol_nb_design, num_components, device="cuda"):
    """Compute the Sobol indices of the model using the Sobol sequence.
    
    Args:
      A: tensor of shape (batch_size, num_features)
      U: tensor of shape (batch_size, num_components)
      W: tensor of shape (num_components, num_features)
      model: Callable
      sobol_nb_design: int, higher reduces variance but increases runtime cost.
      num_components: int, number of concepts in basis.
    """
    masks = ScipySobolSequence()(num_components, nb_design=sobol_nb_design)

    estimator = JansenEstimator()

    importances = []
    assert len(A) == len(U)
    rel_errors_all_masks = []
    for a, u in tqdm.tqdm(zip(A, U), total=len(A)):
      y_pred, rel_errors = concept_perturbation(a, u, masks, W, model, device)
      stis = estimator(masks, y_pred, sobol_nb_design)
      importances.append(stis)
      rel_errors_all_masks.append(rel_errors)

    global_importance = np.mean(importances, 0)
    rel_errors_all_masks = np.concatenate(rel_errors_all_masks)

    return global_importance, rel_errors_all_masks


def sobol_importance_from_sample(features, u, W, model, sampled, num_components, sobol_nb_design, device="cuda"):
  """Sobol indices from a sample of the dataset.
  
  Args:
    features: torch.Tensor of shape (num_samples, num_features)
    u: torch.Tensor of shape (num_samples, num_components)
    W: torch.Tensor of shape (num_components, num_features)
    model: Callable 
    sampled: int, number of sampled to use to compute sobol indices. (higher reduces variance but increases runtime cost)
    num_components: int, number of concepts in basis.
    sobol_nb_design: int, higher reduces variance but increases runtime cost.

  Returns:
    tuple of (global_importance, rel_errors_all_masks)
  """
  a = features.cpu().numpy()[:sampled]
  u_subsampled = u[:sampled]
  return sobol_importance(a, u_subsampled,
                          W, model,
                          sobol_nb_design=sobol_nb_design,
                          num_components=num_components, 
                          device=device)


def crop_concepts(W, criterion, num_or_threshold):
  """Crop concepts with importance below threshold.
  
  Args:
    W: torch.Tensor of shape (num_components, num_features)
    criterion: np.array of shape (num_components,) to assign an importance to each concept.
    num_or_threshold: int or float.
      If int, keep the `num_or_threshold` most important concepts.
      If float, keep concepts with importance above `num_or_threshold`.
  
  Returns:
    a tuple of (W, to_keep) of
      W: torch.Tensor of shape (num_components, num_features)
      to_keep: np.array of shape (num_components,) with True for kept concepts.
  """
  W = W.copy()
  if isinstance(num_or_threshold, int):
    num_to_keep = num_or_threshold
    idx = np.argsort(criterion)[::-1]
    W = W[idx]
    W = W[:num_to_keep]
    to_keep = np.zeros(len(criterion), dtype=bool)
    to_keep[idx[:num_to_keep]] = True
  elif isinstance(num_or_threshold, float):
    to_keep = criterion > num_or_threshold
    W = W[to_keep]
  return W, to_keep



def build_gender_neutral_features(U, W_no_gender, to_keep):
      """
      Build gender neutral features.

        Args:
            U: numpy.array of shape (batch_size, num_components)
            W_no_gender: numpy.array of shape (num_components_cropped, num_features), the product of S and Vt obtained by SVD
            to_keep: np.array of shape (num_components,) with boolean values and num_components_cropped True values.
            num_components: int

        Returns:
            A: torch.Tensor of shape (batch_size, num_features)
      """
      U = U[:, to_keep]  # drop concepts
      A = np.matmul(U, W_no_gender)  # reconstruct the matrix A by multiplying U and the cropped W
      return A