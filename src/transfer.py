import torch
import torch.nn as nn
from torch.nn import init
import pickle

import loader

START_TAG = "<START>"
STOP_TAG = "<STOP>"

output_path = 'models/model_10k_train_ENER_transfer_adapter'

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    init.xavier_normal_(input_linear.weight.data)
    init.normal_(input_linear.bias.data)

# model0 = torch.load("models/test1")
model1 = torch.load("models/model_adapter")
model2 = torch.load("models/model_10k_train_ENER")
model3 = torch.load("models/model_general_iobes_gpu")
#
# print(model1.char_embeds)
model1.char_embeds = model2.char_embeds

# transfer lstm parameters from general to legal
model1.lstm = model3.lstm

# print(model1.char_embeds)
# print(model1.char_cnn3)
model1.char_cnn3 = model2.char_cnn3
# print(model1.char_cnn3)
# print(model1.vocab_size)
model1.vocab_size = model2.vocab_size
# print(model1.vocab_size)
# print(model1.word_embeds)
model1.word_embeds = model2.word_embeds
# print(model1.word_embeds)
#
# print(model1.tagset_size)
model1.tagset_size = model2.tagset_size
# print(model1.tagset_size)
# print(model1.tag_to_ix)
model1.tag_to_ix = model2.tag_to_ix
# print(model1.tag_to_ix)
# print(model1.hidden2tag)
model1.hidden2tag = nn.Linear(model1.hidden_dim * 2, model1.tagset_size)
init_linear(model1.hidden2tag)
# print(model1.hidden2tag)
print(model1.use_crf)

if model1.use_crf:
    print(model1.transitions.shape)
    model1.transitions = nn.Parameter(torch.randn(model1.tagset_size, model1.tagset_size))
    model1.transitions.data[:, model1.tag_to_ix[START_TAG]] = -10000
    model1.transitions.data[model1.tag_to_ix[STOP_TAG], :] = -10000
#     print(model1.transitions)
#
torch.save(model1, output_path)