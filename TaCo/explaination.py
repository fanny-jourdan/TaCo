"""
This module implements the local part of COCKATIEL: occlusion. This allows us to estimate the presence of
concepts in parts of the input text. 
We have modified several parts to adapt the explanation for decomposition methods other than NMF.
"""
import torch
import numpy as np
import sklearn.decomposition
#from nltk.tokenize import word_tokenize
from scipy.optimize import minimize

from flair.models import SequenceTagger
from flair.data import Sentence

from typing import List, Callable, Optional, Union, Tuple

import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import colorsys


def simple_tokenize(text: str) -> list:
    import re
    # Use regex to split on whitespace and common punctuation marks
    return re.findall(r"\w+|[.,!?;]", text)

#plt.style.use('seaborn') 'ggplot'


tagger = SequenceTagger.load("flair/chunk-english")

def extract_clauses(ds_entry: Union[List[str], str], clause_type=['NP', 'ADJP']) -> List[str]:
    """
    Separates the input texts into clauses, and only keeps the ones belonging to the specified types.
    If clause_type is None, the texts are split but all the clauses are kept.

    Parameters
    ----------
    ds_entry
        A list of strings that we wish to separate into clauses.
    clause_type
        A list with the types of clauses to keep. If None, all clauses are kept.

    Returns
    -------
    clause_list
        A list with input texts split into clauses.
    """
    s = Sentence(ds_entry)
    tagger.predict(s)
    clause_list = []
    for segment in s.get_labels():
        if clause_type is None:
            clause_list.append(segment.data_point.text)
        elif segment.value in clause_type:
            clause_list.append(segment.data_point.text)

    return clause_list



def acti_preprocess(activations: torch.Tensor) -> np.ndarray:
    """
    A function to preprocess the activations to work with COCKATIEL
    """
    if len(activations.shape) == 4:
        activations = torch.mean(activations, (1, 2))

    if isinstance(activations, np.ndarray):
        activations = torch.Tensor(activations)

    return activations.cpu().numpy().astype(np.float32)



def calculate_U(A, W):
    """
    Calculate the optimal matrix U such that it minimizes the error between A and U*W.
    
    :param A: np.ndarray, the matrix A of shape (n, p)
    :param W: np.ndarray, the matrix W of shape (r, p)
    :return: np.ndarray, the optimized matrix U of shape (n, r)
    """
    n, p = A.shape  # Extract the shape of matrix A
    r, _ = W.shape  # Extract the number of rows of matrix W 
    # Define the objective function to minimize
    def objective(U_flat):
        U = U_flat.reshape((n, r))  # Reshape U_flat to a matrix of shape (n, r)
        error = np.linalg.norm(A - np.dot(U, W))  # Compute the norm of the error between A and U*W
        return error
    # Initialize U with zeros and flatten it to a 1D array
    U_initial_flat = np.zeros((n, r)).flatten()
    # Perform the optimization
    result = minimize(objective, U_initial_flat, method='L-BFGS-B')
    # Extract the optimized matrix U from the result and reshape it to the correct shape
    U_optimal = result.x.reshape((n, r))
    return U_optimal


def calculate_u_values(sentence, cropped_sentences, model, tokenizer, W,
                       separate, ignore_words: Optional[List[str]] = None, device='cuda') -> np.ndarray:
    if ignore_words is None:
        ignore_words = []
    with torch.no_grad():
        activations = None
        for crop_id in range(-1, len(cropped_sentences)):
            if crop_id == -1:
                perturbated_review = sentence
            elif cropped_sentences[crop_id] not in ignore_words:
                perturbated_review = separate.join(np.delete(cropped_sentences, crop_id))
            else:
                continue
            tokenized_perturbated_review = tokenizer(perturbated_review, truncation=True, padding=True, return_tensors="pt").to(device)
            activation = model.features(**tokenized_perturbated_review)
            activations = activation if activations is None else torch.cat([activations, activation])

        activations = acti_preprocess(activations)

        u_values = calculate_U(activations, W)
        return u_values


def calculate_importance(
        words: List[str], u_values: np.ndarray, concept_id: int, ignore_words: List[str]
) -> List[float]:
    """
    Calculates the presence of concepts in the input list of words.
    """
    u_delta = u_values[0, concept_id] - u_values[1:, concept_id]
    importances = []
    delta_id = 0  # pointer to get current id in importance (as we skip unused word)

    for word_id in range(len(words)):
        if words[word_id] not in ignore_words:
            importances.append(u_delta[delta_id])
            delta_id += 1
        else:
            importances.append(0.0)

    return importances



def occlusion_concepts(
        sentence: str,
        model,
        tokenizer: Callable,
        W: torch.tensor,
        l_concept_id: np.ndarray,
        ignore_words: Optional[List[str]] = None,
        extract_fct: str = "clause",
        device='cuda'
) -> np.ndarray:
    """
    Generates explanations for the input sentence using COCKATIEL.

    It computes the presence of the concepts of interest (in l_concept_id) using the
    W matrix in chosen factorization.

    The granularity of the explanations is set with extract_fct.

    Parameters
    ----------
    sentence
        The string (sentence) we wish to explain using COCKATIEL.
    model
        The model under study.
    tokenizer
        A Callable that transforms strings into tokens capable of being ingested by the model.
    W
        The torch Tensor of the second matrix of the decomposition.
    l_concept_id
        Either a list of concepts of interest (for a given task).
    ignore_words
        A list of strings to ignore when applying occlusion.
    extract_fct
        A string indicating whether at which level we wish to explain: "word", "clause" or "sentence".
    device
        The device on which tensors are stored ("cpu" or "cuda").

    Returns
    -------
    l_importances
        An array with the presence of each concept in the input sentence.
    """
    sentence = str(sentence)

    if extract_fct == "clause":
        words = extract_clauses(sentence, clause_type=None)
        separate = " "

    else:
        words = simple_tokenize(sentence)
        if extract_fct == "sentence":
            separate = ". "
        elif extract_fct == "word":
            separate = " "
        else:
            raise ValueError("Extraction function can be only 'clause', 'sentence', or 'word")

    
    u_values = calculate_u_values(sentence, words,  model, tokenizer, W, separate, ignore_words, device)
    l_importances = []
    for concept_id in l_concept_id:
        importances = calculate_importance(words, u_values, concept_id, ignore_words)
        l_importances.append(np.array(importances))

    return np.array(l_importances)


def viz_concepts(
        text,
        explanation,
        colors,
        ignore_words: Optional[List[str]] = None,
        extract_fct: str = "clause"
):
    """
    Generates the visualization for COCKATIEL's explanations.

    Parameters
    ----------
    text
        A string with the text we wish to explain.
    explanation
        An array that corresponds to the output of the occlusion function.
    ignore_words
        A list of strings to ignore when applying occlusion.
    extract_fct
        A string indicating whether at which level we wish to explain: "word", "clause" or "sentence".
    colors
        A dictionary with the colors for each label
    """
    try:
        text = text.decode('utf-8')
    except:
        text = str(text)

    if extract_fct == "clause":
        words = extract_clauses(text, clause_type=None)
    else:
        words = simple_tokenize(text)

    l_phi = np.array(explanation)

    phi_html = []

    p = 0  # pointer to get current color for the words (it does not color words that have no phi)
    for i in range(len(words)):
        if words[i] not in ignore_words:
            k = 0
            for j in range(len(l_phi)):
                if l_phi[k][p] < l_phi[j][p]:
                    k = j

            if l_phi[k][p] > 0.2:
                phi_html.append(f'<span style="background-color: {colors[k]} {l_phi[k][p]}); padding: 1px 5px; border: solid 3px ; border-color: {colors[k]} 1); #EFEFEF">{words[i]}</span>')
                p += 1
            else:
                phi_html.append(f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF">{words[i]}</span>')
                p += 1
        else:
            phi_html.append(f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF">{words[i]}</span>')
    display(HTML("<div style='display: flex; width: 400px; flex-wrap: wrap'>" +  " ".join(phi_html) + " </div>" ))
    display(HTML('<br><br>'))


def print_legend(l_concept_id, importance_gender, importance_occ, conceptnames = None):
    """
    This function generates and displays a legend where each item represents a concept with a unique color and label.
    It also displays the importance of gender and occupation for each concept using HTML and CSS.
    
    Parameters:
    l_concept_id (list): A list containing the IDs of the concepts to be included in the legend.
    importance_gender (list): A list containing the values are the importance of gender for each concept.
    importance_occ (list): A list containing the values are the importance of occupation for each concept.
    conceptnames (list, optional): A list containing the names of the concepts. If not provided, the concepts will be labeled as Concept1, Concept2, etc.
    
    Returns:
    dict: A dictionary where the keys are concept IDs and the values are the corresponding RGBA color strings.
    """
    n = len(l_concept_id)
    colors = {}
    label_to_criterion = {}
    label_to_criterion2 = {}
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)  # Convertit HSV à RGB
        rgba_string = f"rgba({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)}, "
        colors[i] = rgba_string
        if conceptnames == None:
            label_to_criterion[i] = f"Concept{i + 1}"
        else:
            label_to_criterion[i] = f"Concept{conceptnames[i]}"
        label_to_criterion2[i] = str(importance_gender[l_concept_id[i]]) + " / " + str(importance_occ[l_concept_id[i]])
    for label_id in label_to_criterion.keys():
        html = []
        html.append(f'<span style="background-color: {colors[label_id]} 0.5); padding: 1px 5px; border: solid 3px ; border-color: {colors[label_id]} 1); #EFEFEF">{label_to_criterion[label_id]} </span>')
        html.append(f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF"> <I> {label_to_criterion2[label_id]} </I> </span>')
        display(HTML("<div style='display: flex; width: 400px; flex-wrap: wrap'>" +  " ".join(html) + " </div>" ))
        display(HTML('<br><br>'))
    return(colors)


def print_legend2(colors, l_concept_id, importance_gender, importance_occ, conceptnames = None):
    """
    This function generates and displays a legend where each item represents a concept with a unique color and label.
    It also displays the importance of gender and occupation for each concept using HTML and CSS.
    
    Parameters:
    colors (dict): A dictionary where the keys are concept IDs and the values are the corresponding RGBA color strings.
    l_concept_id (list): A list containing the IDs of the concepts to be included in the legend.
    importance_gender (list): A list containing the values are the importance of gender for each concept.
    importance_occ (list): A list containing the values are the importance of occupation for each concept.
    conceptnames (list, optional): A list containing the names of the concepts. If not provided, the concepts will be labeled as Concept1, Concept2, etc.

    """
    n = len(l_concept_id)
    label_to_criterion = {}
    label_to_criterion2 = {}

    def percent(x):
        x = x * 100
        return(round(x, 2))

    for i in range(n):
        if conceptnames == None:
            label_to_criterion[i] = f"Concept{i + 1}"
        else:
            label_to_criterion[i] = f"Concept{conceptnames[i]}"
        label_to_criterion2[i] = str(percent(importance_gender[l_concept_id[i]])) + "% / " + str(percent(importance_occ[l_concept_id[i]])) + "%"
    for label_id in label_to_criterion.keys():
        html = []
        html.append(f'<span style="background-color: {colors[label_id]} 0.5); padding: 1px 5px; border: solid 3px ; border-color: {colors[label_id]} 1); #EFEFEF">{label_to_criterion[label_id]} </span>')
        html.append(f'<span style="background-color: rgba(233,30,99,0);  padding: 1px 5px; border: solid 3px ; border-color:  rgba(233,30,99,0); #EFEFEF"> <I> {label_to_criterion2[label_id]} </I> </span>')
        display(HTML("<div style='display: flex; width: 400px; flex-wrap: wrap'>" +  " ".join(html) + " </div>" ))
        display(HTML('<br><br>'))