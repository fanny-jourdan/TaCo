# 🌮 TaCo: Targeted Concept Erasure Prevents Non-Linear Classifiers From Detecting Protected Attributes

This repository contains code for the paper:

*TaCo: Targeted Concept Erasure Prevents Non-Linear Classifiers From Detecting Protected Attributes*

The code is implemented and available **for Pytorch**. 

Running example of TaCo method:
```python
from TaCo.TaCo import found_concepts, remove_concept_on_clstoken

method_decompose = "PCA" # Name of the chosen dimensionality reduction method (e.g., PCA)
num_components = 20 # Total number of components in the decomposition (concepts)
sobol_nb_design, sobol_sampled = 50, 10_000 # Hyperparameters for Sobol importance part
nb_cpt_remov = 3 # Number of concepts to be erased

#First steps (run only once)
U_train, U_val, U_test, W, angle = found_concepts(train_clstoken, val_clstoken, test_clstoken,
                                                  model, pg_model, device,
                                                  method_decompose, num_components, 
                                                  sobol_nb_design, sobol_sampled)

#Second steps (run as many times as you like to see how your model evolves with the number of concepts erased )
train_clstoken_no_gender, val_clstoken_no_gender, test_clstoken_no_gender = remove_concept_on_clstoken(U_train, U_val, U_test, W, angle, nb_cpt_remov, num_components) 
  
```
At the end, you have the CLS tokens (or, more generally, the final latent space representation), from which the specified number of concepts has been erased in the order described in the paper.


You can found the paper figures on [visualization notebook](./visualizations.ipynb).


