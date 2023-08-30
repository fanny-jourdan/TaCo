from math import ceil
import tqdm
import torch
from torch import nn
from transformers import RobertaPreTrainedModel, AutoTokenizer, RobertaModel


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

        self.roberta = RobertaModel(config, add_pooling_layer=False)
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
        logits = self.classifier(sequence_output)  # Since classifier's forward method now calls both features and end_model

        return logits



def get_roberta(model_path, num_labels):
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = CustomRobertaForSequenceClassification.from_pretrained(model_path, num_labels)
  model.eval()

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.to(device)

  return model, tokenizer


def batcher(elements, batch_size):
  nb_batchs = ceil(len(elements) / batch_size)

  for batch_i in tqdm.tqdm(range(nb_batchs)):
    batch_start = batch_i * batch_size
    batch_end = batch_start + batch_size

    batch = elements[batch_start:batch_end]
    yield batch


def tokenize(tokenizer, samples, device = 'cuda'):
  samples = [s for s in samples]
  x = tokenizer(samples, padding="max_length",
                max_length = 512, truncation = True,
                return_tensors='pt')
  x = x.to(device)
  return x


def preprocess(tokenizer, samples, device = 'cuda'):
  x, y = samples[:, 0], samples[:, 1]
  x = tokenize(tokenizer, x, device)
  y = torch.Tensor(y.astype(int)).to(device)
  return x, y


def batch_predict(model, tokenizer, inputs, batch_size = 64, device = 'cuda'):
  predictions = None
  labels = None
  with torch.no_grad():
    for batch_input in batcher(inputs, batch_size):
      xp, yp = preprocess(tokenizer, batch_input, device) 
      out_batch = model(**xp)
      predictions = out_batch if predictions is None else torch.cat([predictions, out_batch])
      labels = yp if labels is None else torch.cat([labels, yp])
    
    return predictions, labels
  