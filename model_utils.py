import torch
from torch import nn

from transformers import AutoTokenizer, DebertaV2Tokenizer
from transformers import RobertaForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import DebertaV2ForSequenceClassification

from typing import Optional, List, Tuple, Dict

class DistilBertWrapper(DistilBertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, attention_mask=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["logits"] 

    def features(self, input_ids, attention_mask=None):
        return (self.distilbert(input_ids=input_ids, attention_mask=attention_mask))["last_hidden_state"][:, 0, :]

    def end_model(self, x):
        x = self.pre_classifier(x)
        x = self.classifier(x)
        x = self.dropout(x)
        return x
    

class RobertaWrapper(RobertaForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, attention_mask=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  

    def features(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return (outputs.last_hidden_state)[:,0,:]
    
    def end_model(self, x):
        x = self.classifier.dense(x)
        x = self.classifier.dropout(x)
        x = self.classifier.out_proj(x)
        return x


class DebertaWrapper(DebertaV2ForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, attention_mask=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  

    def features(self, input_ids, attention_mask=None):
        outputs = (self.deberta(input_ids=input_ids, attention_mask=attention_mask)).last_hidden_state
        return self.pooler(outputs)
    
    def end_model(self, x):
        x = self.classifier(x)
        x = self.dropout(x)
        return x
    


def get_model(model_path, model_type = "RoBERTa", num_labels: int = 28):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  
  if model_type == "RoBERTa":
      tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
      model = RobertaWrapper.from_pretrained(model_path)

  elif model_type == "DistilBERT":
      tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
      model = DistilBertWrapper.from_pretrained(model_path)

  elif model_type == "DeBERTa":
      tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
      model = DebertaWrapper.from_pretrained(model_path)
  else:
      return("Error: model_type must be either RoBERTa, DistilBERT or DeBERTa.")

  model.eval()
  model.to(device)
  return model, tokenizer




# class CustomRobertaClassificationHead(nn.Module):
#     def __init__(self, config, num_labels):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         drop_rate = config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.
#         self.dropout = nn.Dropout(drop_rate)
#         self.out_proj = nn.Linear(config.hidden_size, num_labels)

#     def forward(self, features, **kwargs):
#         x = self.features(features) # Use the features method to get the intermediate representation
#         x = self.end_model(x)       # Then, get the final logits with the end_model method
#         return x
    
#     def features(self, x, **kwargs):  
#         x = self.dropout(x)
#         x = self.dense(x)
#         return x
    
#     def end_model(self, x):
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


# class CustomRobertaForSequenceClassification(RobertaPreTrainedModel):
#     _keys_to_ignore_on_load_missing = [r"position_ids"]

#     def __init__(self, config, num_labels):
#         super().__init__(config)
#         self.num_labels = num_labels
#         self.config = config

#         # transformer part
#         self.roberta = RobertaModel(config, add_pooling_layer=False)

#         # two dense layers
#         #output of the first dense layer is considered embeddings
#         self.classifier = CustomRobertaClassificationHead(config, num_labels)

#         self.post_init()
    
#     def features(
#         self,
#         input_ids = None,
#         attention_mask = None,
#         token_type_ids = None,
#         position_ids = None,
#         head_mask = None,
#         inputs_embeds = None,
#         output_attentions = None,
#         output_hidden_states = None,
#         return_dict = None,
#     ):
#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         return self.classifier.features(outputs[0][:, 0, :])

#     def end_model(self, activations):
#         return self.classifier.end_model(activations)

#     def forward(
#         self,
#         input_ids = None,
#         attention_mask = None,
#         token_type_ids = None,
#         position_ids = None,
#         head_mask = None,
#         inputs_embeds = None,
#         output_attentions = None,
#         output_hidden_states = None,
#         return_dict = None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         ) 
#         sequence_output = outputs[0][:, 0, :]
#         logits = self.classifier(sequence_output)
#         return logits


