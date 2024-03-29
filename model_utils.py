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