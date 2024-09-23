from math import ceil
import tqdm
import torch


def batcher(elements, batch_size):
  nb_batchs = ceil(len(elements) / batch_size)

  for batch_i in tqdm.tqdm(range(nb_batchs)):
    batch_start = batch_i * batch_size
    batch_end = batch_start + batch_size

    batch = elements[batch_start:batch_end]
    yield batch


# def tokenize(tokenizer, samples, device = 'cuda'):
#   samples = [s for s in samples]
#   x = tokenizer(samples, padding="max_length",
#                 max_length = 512, truncation = True,
#                 return_tensors='pt')
#   x = x.to(device)
#   return x

def tokenize(tokenizer, samples, device = 'cuda'):
    samples = [s for s in samples]
    x = tokenizer(samples, padding="max_length",
                  max_length = 512, truncation = True,
                  return_tensors='pt')
    x = {key: val.to(device) for key, val in x.items() if key in ["input_ids", "attention_mask"]}
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