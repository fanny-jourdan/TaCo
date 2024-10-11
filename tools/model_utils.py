import torch
from torch import nn

from transformers import AutoTokenizer, DebertaV2Tokenizer
from transformers import RobertaForSequenceClassification
from transformers import DistilBertForSequenceClassification
from transformers import DebertaV2ForSequenceClassification
from transformers import T5ForSequenceClassification
from transformers import LlamaForSequenceClassification


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
    



class T5Wrapper(T5ForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            **kwargs
        )
        return outputs.logits

    def features(self, input_ids=None, attention_mask=None, **kwargs):
        # Vérifier si input_ids est fourni
        if input_ids is None:
            raise ValueError("Input_ids must be provided.")

        # Gérer decoder_input_ids
        decoder_input_ids = kwargs.get('decoder_input_ids', None)
        decoder_inputs_embeds = kwargs.get('decoder_inputs_embeds', None)
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(input_ids)

        # Obtenir les sorties de l'encodeur
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=kwargs.get('decoder_attention_mask', None),
            head_mask=kwargs.get('head_mask', None),
            decoder_head_mask=kwargs.get('decoder_head_mask', None),
            cross_attn_head_mask=kwargs.get('cross_attn_head_mask', None),
            encoder_outputs=kwargs.get('encoder_outputs', None),
            inputs_embeds=kwargs.get('inputs_embeds', None),
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=kwargs.get('use_cache', False),
            output_attentions=kwargs.get('output_attentions', None),
            output_hidden_states=kwargs.get('output_hidden_states', None),
            return_dict=True,
        )
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)

        # Créer le masque EOS
        eos_mask = input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")

        # Extraire la représentation de la phrase
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]

        return sentence_representation

    def end_model(self, x):
        logits = self.classification_head(x)
        return logits




class LlamaWrapper(LlamaForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]  # last_hidden_state de taille (batch_size, seq_length, hidden_size)

        # Pooling : sélection du premier token (par convention pour la classification)
        pooled_output = hidden_states[:, 0, :]

        # Application de la couche de classification
        logits = self.score(pooled_output)
        return logits

    def features(self, input_ids, attention_mask=None):
        # Reproduire exactement les mêmes étapes que dans forward jusqu'à pooled_output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs[0]

        # Même pooling que dans forward
        pooled_output = hidden_states[:, 0, :]

        return pooled_output

    def end_model(self, x):
        # Application de la même couche de classification que dans forward
        logits = self.score(x)
        return logits


def get_model(model_path, model_type = "RoBERTa"):
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

  elif model_type == "t5":
      tokenizer = AutoTokenizer.from_pretrained("t5-small")
      model = T5Wrapper.from_pretrained(model_path)
      
  elif model_type == "Llama3":
      tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
      model = LlamaWrapper.from_pretrained(model_path)
      
  else:
      return("Error: model_type must be either RoBERTa, DistilBERT, DeBERTa, T5 or Llama3.")

  model.eval()
  model.to(device)
  return model, tokenizer