import numpy as np
import matplotlib.pyplot as plt
from datasets_tools import get_occupation_names, get_gender_names
#from fairnessmetrics import get_named_metrics


def plot_co_occurence(co_occurrence_matrix, per_class=True):
  # Set up the figure and axes
  fig, ax = plt.subplots(figsize=(10, 10))

  if per_class:
    co_occurrence_matrix = co_occurrence_matrix / co_occurrence_matrix.sum(axis=1, keepdims=True) * 100
  else:
    co_occurrence_matrix = co_occurrence_matrix / co_occurrence_matrix.sum() * 100

  # Create a heatmap using imshow
  im = ax.imshow(co_occurrence_matrix, cmap='winter', origin='lower', extent=[0, 4, 0., 28])

  gender_names = get_gender_names()
  occ_names = get_occupation_names()

  # Set ticks and tick labels
  ax.set_xticks([2., 4.])
  ax.set_yticks(np.arange(28))
  ax.set_xticklabels(gender_names)
  ax.set_yticklabels(occ_names)

  # Rotate the x-labels and set the axis labels
  plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
  ax.set_xlabel("Gender")
  ax.set_ylabel("Occupation")

  # Loop over the data and create annotations
  for i in range(28):
    text = ax.text(1, i + 0.5, f"{co_occurrence_matrix[i, 0]:.1f}%",
                  ha="center", va="center", color="w", fontweight='bold')
    text = ax.text(3, i + 0.5, f"{co_occurrence_matrix[i, 1]:.1f}%",
                  ha="center", va="center", color="w", fontweight='bold')

  # Set a title for the plot
  ax.set_title("Co-occurrence Matrix of Occupations and Gender")

  # Add a colorbar
  cbar = ax.figure.colorbar(im, ax=ax)
  cbar.ax.set_ylabel("Percentage", rotation=-90, va="bottom")

  # Adjust the layout and display the plot
  plt.tight_layout()
  plt.show()


def plot_co_importance(global_importance_occ, global_importance_gender):
  colors = np.arange(len(global_importance_gender))
  angle = np.arctan(global_importance_occ / global_importance_gender) * 180 / np.pi
  colors = 90. - angle
  plt.scatter(global_importance_gender, global_importance_occ, c=colors, cmap='viridis', s=70, alpha=0.7)
  plt.xlabel('Importance for gender')
  plt.ylabel('Importance for occupation')
  plt.title('Concept importance before concept cropping')
  cbar = plt.colorbar()
  cbar.set_label('Angle (°).')
  plt.grid(True)
  #plt.legend()
  plt.show()


def plot_reconstruction_error(rel_errors_all_masks):
  import seaborn as sns
  sns.histplot(np.random.choice(rel_errors_all_masks, size=10_000), stat='probability')
  plt.xlabel('Relative error')
  plt.ylabel('Probability')
  plt.title('Reconstruction errors with $M=800$ masks.')



