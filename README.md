# 🌮 TaCo: Targeted Concept Removal in Output Embeddings for NLP via Information Theory and Explainability

This repository contains code for the paper:

*TaCo: Targeted Concept Removal in Output Embeddings for NLP via Information Theory and Explainability*

The code is implemented and available **for Pytorch**. 

Running example of TaCo method:
```python
from TaCo.TaCo import found_concepts, remove_concept_on_clstoken

method_decompose = "SVD"
num_components = 20
sobol_nb_design = 50
sobol_sampled = 10_000
nb_cpt_remov = 3

#First steps (run only once)
U_train, U_val, U_test, W, angle = found_concepts(train_clstoken, val_clstoken, test_clstoken,
                                                  model, pg_model, device,
                                                  method_decompose, num_components, 
                                                  sobol_nb_design, sobol_sampled)

#Second steps (run as many times as you like to see how your model evolves with the number of concepts removed )
train_clstoken_no_gender, val_clstoken_no_gender, test_clstoken_no_gender = remove_concept_on_clstoken(U_train, U_val, U_test, W, angle, nb_cpt_remov, num_components) 
  
```
At the end, you have the CLS tokens, from which the number of concepts requested have been removed (in the order explained in the paper).

A detailed notebook (step by step) is available: [notebook example](./example.ipynb).

You can found the paper figures on [visualization notebook](./visualization.ipynb).


