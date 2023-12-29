import torch
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel, AutoTokenizer
from transformers import DebertaV2PreTrainedModel, DebertaV2Model, DebertaV2Tokenizer, DebertaV2ForSequenceClassification

class CustomRobertaClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.features(features) # Use the features method to get the intermediate representation
        x = self.end_model(x)       # Then, get the final logits with the end_model method
        return x
    
    def features(self, x, **kwargs):  
        x = self.dropout(x)
        x = self.dense(x)
        return x
    
    def end_model(self, x):
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CustomRobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.config = config

        # transformer part
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # two dense layers
        #output of the first dense layer is considered embeddings
        self.classifier = CustomRobertaClassificationHead(config, num_labels)

        self.post_init()
    
    def features(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return self.classifier.features(outputs[0][:, 0, :])

    def end_model(self, activations):
        return self.classifier.end_model(activations)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) 
        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output)
        return logits

class CustomDebertaV2ClassificationHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.features(features)
        x = self.end_model(x)
        return x

    def features(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        return x

    def end_model(self, x):
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomDebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
#class CustomDebertaV2ForSequenceClassification(DebertaV2ForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.config = config

        # Transformer part
        self.deberta = DebertaV2Model(config)
        # Custom classification head
        self.classifier = CustomDebertaV2ClassificationHead(config, num_labels)

        self.post_init()

    def features(self, **kwargs):
        outputs = self.deberta(**kwargs)
        return self.classifier.features(outputs[0][:, 0, :])

    def end_model(self, activations):
        return self.classifier.end_model(activations)

    def forward(self, **kwargs):
        return_dict = kwargs.get('return_dict', self.config.use_return_dict)

        outputs = self.deberta(**kwargs)
        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output)
        return logits

def adjust_weight_names_DeBERTa(pretrained_state_dict):
    weight_map = {
        'pooler.dense.weight': 'classifier.dense.weight',
        'pooler.dense.bias': 'classifier.dense.bias',
        'classifier.bias': 'classifier.out_proj.bias',
        'classifier.weight': 'classifier.out_proj.weight'
    }

    adjusted_state_dict = {}
    for key in pretrained_state_dict:
        new_key = weight_map.get(key, key)  # Utiliser la nouvelle clé si elle existe dans weight_map
        adjusted_state_dict[new_key] = pretrained_state_dict[key]
    
    return adjusted_state_dict


def get_model(model_path, model_type = "RoBERTa", num_labels: int = 28):
  if model_type == "RoBERTa":
      tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
      model = CustomRobertaForSequenceClassification.from_pretrained(model_path, num_labels, local_files_only=True)

  elif model_type == "DeBERTa":
      tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
      # Initialisation du modèle personnalisé
      pretrained_model = DebertaV2ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, local_files_only=True)
      adjusted_state_dict = adjust_weight_names_DeBERTa(pretrained_model.state_dict())
      
      model = CustomDebertaV2ForSequenceClassification(pretrained_model.config, num_labels=num_labels)
      model.load_state_dict(adjusted_state_dict)
      #model = CustomDebertaV2ForSequenceClassification.from_pretrained(model_path, num_labels, local_files_only=True)
        
  else:
      return("Error: model_type must be either RoBERTa or DeBERTa.")

  model.eval()
  #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  #model.to(device)
  return model, tokenizer