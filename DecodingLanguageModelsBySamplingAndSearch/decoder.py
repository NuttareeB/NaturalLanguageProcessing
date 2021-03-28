import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchtext.datasets import LanguageModelingDataset
from torchtext.data import SubwordField, Field, BPTTIterator
from torchtext.data import get_tokenizer
import sys
import argparse
from LanguageModel import LanguageModel
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

  

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--chkpt', dest='chkpt', metavar='c', default="got_language_model")   
  args = parser.parse_args()


  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info('Using device: {}'.format(dev))

  logging.info("Loading tokenizer and vocab from vocab.pkl")  
  text_field = pickle.load(open("vocab.pkl", "rb"))
  vocab_size = len(text_field.vocab.itos)

  logging.info("Loading checkpoint {}".format(args.chkpt))
  lm = LanguageModel(vocab_size).to(dev)
  lm.load_state_dict(torch.load(args.chkpt, map_location=dev))
  lm.eval()


  p = "the night is dark and full of terrors"
  
  # Torch is a bit frustrating at times and some things that ought to be deterministic are not. 
  # This is an attempt to resolve that, but it doesn't work 100% of the time
  torch.set_deterministic(True)
  seed = 42
  mlen = 150

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Vanilla Sampling -----------")
  print(sample(lm, text_field, prompt=p, max_len=mlen))
  
  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 0.0001 -------")
  print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))
  
  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 100 --------")
  print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))
  
  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))
  
  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 20 -----------")
  print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))
  
  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.001 -----------")
  print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.75 -----------")
  print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=1 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=10 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=50 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

  print()



############################################################################################
# TASK 1.1
############################################################################################

def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
  tensor = text_field.process([text_field.tokenize(prompt.lower())])
  output = tensor

  hidden_size = model._modules['rnn'].hidden_size
  num_layers = model._modules['rnn'].num_layers
  hidden_state = torch.zeros(num_layers, 1, hidden_size)
  cell_state = torch.zeros(num_layers, 1, hidden_size)
  
  criterian = nn.LogSoftmax(dim=1)

  output, hidden_state, cell_state = model(output, hidden_state, cell_state)
  probs = criterian(output[-1])
  hidden_state = torch.cat([hidden_state]*beams, 1)
  cell_state = torch.cat([cell_state]*beams, 1)
  
  topk_probs, topk_idxs = torch.topk(probs, beams)

  decodedStrings = topk_idxs.view(beams, 1)

  last_prop = []
  for i in range(max_len-1):
    output, ht, ct = model(topk_idxs, hidden_state, cell_state)

    probs = criterian(output[-1])

    # print("probs")
    # print(probs)
    # print("top k")
    # print(topk_probs.view((beams, 1)))
    cum_log_probs = probs + topk_probs.view((beams, 1))

    # print("cum_log_probs k")
    # print(cum_log_probs)
    topk_probs, topk_idxs = torch.topk(cum_log_probs.view(-1), beams)
    beam_index = np.array(np.unravel_index(topk_idxs.numpy(), cum_log_probs.shape)).T

    new_ht = []
    new_ct = []
    for l in range(num_layers):
      ht_layer = []
      ct_layer = []
      for r, c in beam_index:
        ht_layer.append(ht[l][r])
        ct_layer.append(ct[l][r])
      new_ht.append(torch.stack(ht_layer))
      new_ct.append(torch.stack(ct_layer))

    hidden_state = torch.stack(new_ht)
    cell_state = torch.stack(new_ct)

    strs = []
    for i, (r, c) in enumerate(beam_index):
      topk_idxs[i] = c
      strs.append(torch.cat([decodedStrings[r], torch.tensor([c])]))

    decodedStrings = strs
    topk_idxs = topk_idxs.unsqueeze(0)
    last_prop = topk_probs

  decodedStrings = decodedStrings[last_prop.argmax()]
  decodedString = prompt + " " + reverseNumeralize(decodedStrings, text_field)
  return decodedString

############################################################################################
# TASK 1.2
############################################################################################

def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
  assert (k==0 or p==1), "Cannot combine top-k and top-p sampling"

  tensor = text_field.process([text_field.tokenize(prompt.lower())])
  output = tensor

  hidden_size = model._modules['rnn'].hidden_size
  num_layers = model._modules['rnn'].num_layers
  hidden_state = torch.zeros(num_layers, 1, hidden_size)
  cell_state = torch.zeros(num_layers, 1, hidden_size)
  
  criterian = nn.Softmax(dim=1)
  strings=[]

  for i in range(max_len):
    output, ht, ct = model(output, hidden_state, cell_state)
    hidden_state = ht
    cell_state = ct
    probs = criterian(output[-1]/temp)

    if k != 0:
      topk_probs, topk_idxs = torch.topk(probs, k)
      topk_probs = topk_probs/topk_probs.sum(1)
      probs_dist = torch.distributions.Categorical(topk_probs)
      index_prop = probs_dist.sample()
      output = topk_idxs.view(-1)[index_prop].unsqueeze(0)
    elif p > 0:
      sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
      cum_probs = torch.cumsum(sorted_probs, dim=1)
      cum_probs = cum_probs.squeeze()

      idx_reset_probs = cum_probs >= p
      idx_reset_probs[1:] = idx_reset_probs[:-1].clone()
      
      # for the case that every element is True (We only need the highest probability)
      idx_reset_probs[0] = False 

      probs.squeeze()[sorted_idxs[idx_reset_probs.unsqueeze(0)]] = 0

      probs = probs/probs.sum()
      probs_dist = torch.distributions.Categorical(probs)
      output = probs_dist.sample().unsqueeze(0)

    else:
      probs_dist = torch.distributions.Categorical(probs)
      output = probs_dist.sample().unsqueeze(0)

    strings.append(output)
  decodedString = prompt + " " + reverseNumeralize(torch.cat(strings).squeeze(), text_field)
  return decodedString

############################################################################################

def reverseNumeralize(numeralized_string, text_field):
  strings = [text_field.vocab.itos[i] for i in numeralized_string]
  return " ".join(strings)

if __name__ == "__main__":
  main()