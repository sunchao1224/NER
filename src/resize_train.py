import random
from collections import Counter

def read_file(file_path):
    sentences, labels = [], []
    sentence, label = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line or line == "-DOCSTART- O":
                if sentence and label:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
                continue
            token, tag = line.rsplit(' ', 1)
            sentence.append(token)
            label.append(tag)
    if sentence and label:  # left over sentence
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels


def write_to_file(sentences, labels, file_path):
    with open(file_path, 'w') as f:
        for sentence, label in zip(sentences, labels):
            for token, tag in zip(sentence, label):
                f.write(f'{token} {tag}\n')
            f.write('\n')  # new line at the end of each sentence


# def sample_tokens(sentences, labels, num_tokens=2000):
#     token_counter = Counter()
#     for sentence in sentences:
#         token_counter.update(sentence)
#     tokens, distribution = zip(*token_counter.most_common())
#     sampled_sentences, sampled_labels = [], []
#     sampled_tokens = 0
#     while sampled_tokens < num_tokens:
#         sentence_idx = random.choices(range(len(sentences)), weights=distribution)[0]
#         sampled_sentences.append(sentences[sentence_idx])
#         sampled_labels.append(labels[sentence_idx])
#         sampled_tokens += len(sentences[sentence_idx])
#     return sampled_sentences, sampled_labels

def sample_tokens(sentences, labels, num_tokens=20000):
    distribution = [len(sentence) for sentence in sentences]
    sampled_sentences, sampled_labels = [], []
    sampled_tokens = 0
    while sampled_tokens < num_tokens:
        sentence_idx = random.choices(range(len(sentences)), weights=distribution)[0]
        sampled_sentences.append(sentences[sentence_idx])
        sampled_labels.append(labels[sentence_idx])
        sampled_tokens += len(sentences[sentence_idx])
    return sampled_sentences, sampled_labels



# Read your file
file_path = "data/E-NER-Dataset-main/E-NER_train.txt"
sentences, labels = read_file(file_path)

# Sample tokens
sampled_sentences, sampled_labels = sample_tokens(sentences, labels, 5000)

# Write sampled tokens and labels to a file
output_file_path = "data/E-NER-Dataset-main/E-NER_train_5000.txt"
write_to_file(sampled_sentences, sampled_labels, output_file_path)

