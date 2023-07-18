import torch
import torch.nn as nn
from torch.nn import init
import pickle

import loader

#
# with open('models/mapping.pkl', 'rb') as f:
#     mapping = pickle.load(f)

START_TAG = "<START>"
STOP_TAG = "<STOP>"


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    init.xavier_normal_(input_linear.weight.data)
    init.normal_(input_linear.bias.data)

# model0 = torch.load("models/test1")
model1 = torch.load("models/test")
# model2 = torch.load("models/model_full_trained_on_ENER")
# model3 = torch.load("models/model_general_iobes_gpu")

print(model1)
# #
# # print(model1.char_embeds)
# model1.char_embeds = model2.char_embeds
#
# # transfer lstm parameters from general to legal
# model1.lstm = model3.lstm
#
# # print(model1.char_embeds)
# # print(model1.char_cnn3)
# model1.char_cnn3 = model2.char_cnn3
# # print(model1.char_cnn3)
# # print(model1.vocab_size)
# model1.vocab_size = model2.vocab_size
# # print(model1.vocab_size)
# # print(model1.word_embeds)
# model1.word_embeds = model2.word_embeds
# # print(model1.word_embeds)
# #
# # print(model1.tagset_size)
# model1.tagset_size = model2.tagset_size
# # print(model1.tagset_size)
# # print(model1.tag_to_ix)
# model1.tag_to_ix = model2.tag_to_ix
# # print(model1.tag_to_ix)
# # print(model1.hidden2tag)
# model1.hidden2tag = nn.Linear(model1.hidden_dim * 2, model1.tagset_size)
# init_linear(model1.hidden2tag)
# # print(model1.hidden2tag)
# print(model1.use_crf)
#
# if model1.use_crf:
#     print(model1.transitions.shape)
#     model1.transitions = nn.Parameter(torch.randn(model1.tagset_size, model1.tagset_size))
#     model1.transitions.data[:, model1.tag_to_ix[START_TAG]] = -10000
#     model1.transitions.data[model1.tag_to_ix[STOP_TAG], :] = -10000
# #     print(model1.transitions)
# #
# torch.save(model1, 'models/model_adapter_transfer_gpu4_random_outlayer')

# sentences = []
# sentence = []
# for line in open('data/E-NER-Dataset-main/all.csv', 'r', encoding='utf-8'):
#     line = line.rstrip()
#     if not line.split(',')[0] and len(sentence) > 0:
#         # if 'DOCSTART' not in sentence[0][0]:
#         sentences.append(sentence)
#         sentences.append([])
#         sentence = []
#     else:
#         word = line.split(',')
#         assert len(word) >= 2
#         sentence.append(word)
# # if len(sentence) > 0:
# #     if 'DOCSTART' not in sentence[0][0]:
# #         sentences.append(sentence)
# sentences
#
# with open("data/E-NER-Dataset-main/all.txt", 'w') as file:
#     for row in sentences:
#         if not row:
#             file.write('\n')
#         else:
#             for word in row:
#                 s = " ".join(word)
#                 file.write(s + '\n')
# num_per =0
# num_org =0
# num_loc =0
# num_o =0
# num_all = 0
# num_misc =0
# for line in open('data/eng.train', 'r', encoding='utf-8'):
#     line = line.rstrip()
#     if line:
#         tag = line.split()[-1]
#         num_all+=1
#         if "ORG" in tag:
#             num_org+=1
#         elif "PER" in tag:
#             num_per+=1
#             if tag != 'I-PER':
#                 print(tag)
#         elif "LOC" in tag:
#             num_loc+=1
#         elif "MISC" in tag:
#             num_misc+=1
#         elif "O" in tag:
#             num_o+=1
# print("num_all:", num_all)
# print("num_per:", num_per, ":", num_per/num_all)
# print("num_org:", num_org, ":", num_org/num_all)
# print("num_loc:", num_loc, ":", num_loc/num_all)
# print("num_misc:", num_misc, ":", num_misc/num_all)
# print("num_o:", num_o, ":", num_o/num_all)
#


# with open("data/E-NER-Dataset-main/all_tag_changed.txt", 'w') as file:
#     for line in open('data/E-NER-Dataset-main/all.txt', 'r', encoding='utf-8'):
#         line = line.rstrip()
#         if not line:
#             file.write('\n')
#         else:
#             l = line.replace(" B-PERSON", ' B-PER').replace(" I-PERSON", ' I-PER')
#             l = l.replace(" B-BUSINESS", ' B-ORG').replace(" I-BUSINESS", ' I-ORG')
#             l = l.replace(" B-LOCATION", ' B-LOC').replace(" I-LOCATION", ' I-LOC')
#             l = l.replace(" B-MISCELLANEOUS", ' B-MISC').replace(" I-MISCELLANEOUS", ' I-MISC')
#             file.write(l + '\n')

# Find an extra I_LOC in all.txt, finds out that 37899 of all.txt is I-LOC rather than I-LOCATION

# with open('data/E-NER-Dataset-main/all.txt', 'r', encoding='utf-8') as file0:
#     file0 = file0.readlines()
#     for i, line in enumerate(open("data/E-NER-Dataset-main/all_tag_changed.txt", 'r')):
#         if line:
#             line = line.rstrip()
#             if line.endswith(" I-LOC"):
#                 if line.replace(" I-LOC", ' I-LOCATION') != file0[i].rstrip():
#                     print(i)

# with open('data/E-NER-Dataset-main/ENER_train50000.txt', 'w') as train_file:
#     for i, line in enumerate(open('data/E-NER-Dataset-main/ENER_train.txt', 'r')):
#
#         train_file.write(line)
#         if i == 50050:
#             break
#
# def load_sentences(path, lower, zeros):
#     """
#     Load sentences. A line must contain at least a word and its tag.
#     Sentences are separated by empty lines.
#     """
#     sentences = []
#     sentence = []
#     for line in open(path, 'r', encoding='utf-8'):
#         # digit normalization leads to better performance, as the model will be less likely to overfit on specific digit patterns
#         line = line.rstrip()
#         if not line and len(sentence) > 0:
#             if 'DOCSTART' not in sentence[0][0]:
#                 sentences.append(sentence)
#             sentence = []
#         else:
#             word = line.split()
#             print(word)
#             assert len(word) >= 2
#             sentence.append(word)
#     if len(sentence) > 0:
#         if 'DOCSTART' not in sentence[0][0]:
#             sentences.append(sentence)
#     return sentences
#
# path = "data/E-NER-Dataset-main/ENER_train.txt"
# s=load_sentences(path, 1, 1)

# import re
#
#
# def revert_space_to_comma(text):
#     return re.sub(r'(\d) (\d)', r'\1,\2', text)
#
#
# with open("data/E-NER-Dataset-main/E-NER_dev.txt", 'w') as file:
#     for line in open("data/E-NER-Dataset-main/ENER_dev.txt", 'r'):
#         line = line.rstrip()
#         if not line:
#             file.write('\n')
#         else:
#             s = revert_space_to_comma(line)
#             file.write(s + '\n')

