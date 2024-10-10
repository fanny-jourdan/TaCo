import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

from tools.datasets_tools import get_occupation_names, get_gender_names

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
  sns.histplot(np.random.choice(rel_errors_all_masks, size=10_000), stat='probability')
  plt.xlabel('Relative error')
  plt.ylabel('Probability')
  plt.title('Reconstruction errors with $M=800$ masks.')



def prepare_data_drop(methods_data, baseline1_data, drop_metric='Gender'):
    """
    Prepare a DataFrame containing data for plotting accuracy vs. accuracy drop.
    """
    data_list = []

    if drop_metric == 'Gender':
        baseline_accuracy = baseline1_data['Gender Accuracy']
        drop_label = 'Gender Accuracy Drop'
        other_label = 'Occupation Accuracy'
    elif drop_metric == 'Occupation':
        baseline_accuracy = baseline1_data['Occupation Accuracy']
        drop_label = 'Occupation Accuracy Drop'
        other_label = 'Gender Accuracy'
    else:
        raise ValueError("drop_metric must be either 'Gender' or 'Occupation'")

    for method, accuracies in methods_data.items():
        occ_accuracy = accuracies['Occupation Accuracy']
        gender_accuracy = accuracies['Gender Accuracy']

        if drop_metric == 'Gender':
            accuracy_drop = [(baseline_accuracy - acc) / baseline_accuracy for acc in gender_accuracy]
            other_accuracy = occ_accuracy
        else:  # drop_metric == 'Occupation'
            accuracy_drop = [(baseline_accuracy - acc) / baseline_accuracy for acc in occ_accuracy]
            other_accuracy = gender_accuracy

        for drop, other_acc in zip(accuracy_drop, other_accuracy):
            data_list.append({
                'Method': method,
                drop_label: drop,
                other_label: other_acc
            })

    df = pd.DataFrame(data_list)
    return df



def plot_accuracy_vs_accuracy_drop(
    x_col,
    y_col,
    df,
    additional_points=None,
    method_colors=None,
    method_markers=None,
    legend=False,
    xlim=None,
    ylim=None,
    log_scale_x=False,
    title=None,
    min_drop_line=None,
    min_drop_label=None):
    """
    Generalized function to plot accuracy vs accuracy drop.

    Parameters:
    - x_col: Column name for the x-axis.
    - y_col: Column name for the y-axis.
    - df: DataFrame containing the data to plot.
    - additional_points: List of dicts with 'Method', x_col, and y_col.
    - method_colors: Dict mapping method names to colors.
    - method_markers: Dict mapping method names to marker styles.
    - legend: Bool indicating whether to display legend.
    - xlim: Tuple specifying x-axis limits (min, max).
    - ylim: Tuple specifying y-axis limits (min, max).
    - log_scale_x: Bool indicating whether to use a logarithmic scale for the x-axis.
    - title: Title for the plot (if applicable).
    - min_drop_line: Tuple (value, axis) to specify the line position and orientation ('x' for vertical, 'y' for horizontal).
    - min_drop_label: Label for the vertical line (if applicable).
    """
    plt.figure(figsize=(12, 8), dpi=600)

    # Plot data for each method
    for method in df['Method'].unique():
        method_df = df[df['Method'] == method]
        label_method = method if legend else None
        marker_style = method_markers.get(method, 'o') if method_markers else 'o'
        linestyle = '-' #if 'TaCo' not in method else '-'

        if method == 'TaCo PCA':
            zorder = 4  
        elif 'TaCo' in method:
            zorder = 3  
        else:
            zorder = 2

        # Line plot
        sns.lineplot(linewidth=2.5,
            data=method_df,
            x=x_col,
            y=y_col,
            color=method_colors.get(method, 'black'),
            linestyle=linestyle,
            markersize=20,
            label=label_method,
            legend=False,
            zorder=zorder
        )

        # Scatter plot
        if marker_style is not None:
            sns.scatterplot(
                data=method_df,
                x=x_col,
                y=y_col,
                color=method_colors.get(method, 'black'),
                s=100,
                marker=marker_style,
                label=None,
                legend=False,
                zorder=zorder
            )

    # Plot additional points
    if additional_points:
        for point in additional_points:
            method = point['Method']
            x = point[x_col]
            y = point[y_col]
            marker_style = method_markers.get(method, 'o') if method_markers else 'o'
            label_method = method if legend else None

            plt.scatter(
                [x],
                [y],
                s=220,
                marker=marker_style,
                color=method_colors.get(method, 'black'),
                label=label_method,
                zorder=3
            )

    # Add vertical line at min_drop_line if provided
    if min_drop_line is not None:
        line_value, axis = min_drop_line
        if axis == 'x' and (xlim is None or (xlim[0] <= line_value <= xlim[1])):
            plt.axvline(
                x=line_value,
                ymin=0,
                ymax=1,
                color=method_colors.get('Min Line', 'green'),
                linestyle='--',
                linewidth=2,
                label=min_drop_label if legend else None,
                zorder=1
            )
        elif axis == 'y' and (ylim is None or (ylim[0] <= line_value <= ylim[1])):
            plt.axhline(
                y=line_value,
                xmin=0,
                xmax=1,
                color=method_colors.get('Min Line', 'green'),
                linestyle='--',
                linewidth=2,
                label=min_drop_label if legend else None,
                zorder=1
            )

    # Set axis labels
    plt.xlabel(x_col, fontsize=15)
    plt.ylabel(y_col, fontsize=15)

    # Set axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # Set x-axis to logarithmic scale if specified
    if log_scale_x:
        plt.xscale('log')

    # Set title if provided
    if title:
        plt.title(title, fontsize=18)

    # Add legend
    if legend:
        plt.legend(title='Methods', loc='best', fontsize=15)

    plt.tick_params(axis='both', which='major', labelsize=15)

    plt.show()


def prepare_accuracy_data(concept_removed, methods_data, baseline2_data, min_gender):
    """
    Prepare a DataFrame containing all necessary data for the accuracy plots.

    methods_data: dict of method names to dicts with 'Occupation Accuracy' and 'Gender Accuracy' lists.
    baseline2_data: dict with 'Occupation Accuracy' and 'Gender Accuracy' single values.
    min_gender: float, value of min_gender.
    """
    data_list = []
    num_points = len(concept_removed)

    for method, accuracies in methods_data.items():
        occ_accuracy = accuracies['Occupation Accuracy']
        gender_accuracy = accuracies['Gender Accuracy']
        for n, acc in zip(concept_removed, occ_accuracy):
            data_list.append({'Method': method, 'Number of concepts removed': n, 'Accuracy': acc, 'Metric': 'Occupation'})
        for n, acc in zip(concept_removed, gender_accuracy):
            data_list.append({'Method': method, 'Number of concepts removed': n, 'Accuracy': acc, 'Metric': 'Gender'})

    # Baseline 2 data
    occ_accuracy_baseline2 = [baseline2_data['Occupation Accuracy']] * num_points
    gender_accuracy_baseline2 = [baseline2_data['Gender Accuracy']] * num_points
    method = 'Baseline 2'
    for n, acc in zip(concept_removed, occ_accuracy_baseline2):
        data_list.append({'Method': method, 'Number of concepts removed': n, 'Accuracy': acc, 'Metric': 'Occupation'})
    for n, acc in zip(concept_removed, gender_accuracy_baseline2):
        data_list.append({'Method': method, 'Number of concepts removed': n, 'Accuracy': acc, 'Metric': 'Gender'})

    # Min Gender line for Gender Metric
    for n in concept_removed:
        data_list.append({'Method': 'Min Gender', 'Number of concepts removed': n, 'Accuracy': min_gender, 'Metric': 'Gender'})

    df = pd.DataFrame(data_list)
    return df



def plot_accuracy(df, method_colors, concept_removed, legend=False):
    """
    Plot the Occupation and Gender accuracies.
    """
    # Create the palette for seaborn
    palette = {method: color for method, color in method_colors.items() if method in df['Method'].unique()}

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top plot for Occupation
    sns.lineplot(data=df[df['Metric'] == 'Occupation'], x="Number of concepts removed", y="Accuracy", hue="Method",
                 palette=palette, ax=ax1, linewidth=2.5)

    # Bottom plot for Gender
    sns.lineplot(data=df[df['Metric'] == 'Gender'], x="Number of concepts removed", y="Accuracy", hue="Method",
                 palette=palette, ax=ax2, linewidth=2.5)

    # Adjust legend
    if legend:
        handles = [Line2D([], [], color=color, label=method) for method, color in method_colors.items()]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=1, frameon=True)

    # Remove individual legends
    ax1.get_legend().remove()
    ax2.get_legend().remove()

    # Add labels
    ax1.set_ylabel('Occupation Accuracy')
    ax2.set_ylabel('Gender Accuracy')
    ax2.set_xlabel('Number of concepts removed')

    # Set x-axis limits and ticks
    ax1.set_xlim(min(concept_removed), max(concept_removed))
    ax2.set_xlim(min(concept_removed), max(concept_removed))
    ax2.set_xticks(concept_removed)
    ax1.set_xticks(concept_removed)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()