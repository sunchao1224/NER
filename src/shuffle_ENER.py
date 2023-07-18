import random

def read_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def split_data(content, train_ratio=0.66, dev_ratio=0.16, test_ratio=0.16):
    documents = content.split('-DOCSTART- O\n')
    random.shuffle(documents)

    train_count = int(len(documents) * train_ratio)
    dev_count = int(len(documents) * dev_ratio)

    train_data = documents[:train_count]
    dev_data = documents[train_count:train_count + dev_count]
    test_data = documents[train_count + dev_count:]

    return train_data, dev_data, test_data

def write_split_data(output_prefix, train_data, dev_data, test_data):
    with open(output_prefix + '_train.txt', 'w') as train_file:
        train_file.write('-DOCSTART- O\n'.join(train_data))

    with open(output_prefix + '_dev.txt', 'w') as dev_file:
        dev_file.write('-DOCSTART- O\n'.join(dev_data))

    with open(output_prefix + '_test.txt', 'w') as test_file:
        test_file.write('-DOCSTART- O\n'.join(test_data))

def main(input_file, output_prefix):
    content = read_data(input_file)
    train_data, dev_data, test_data = split_data(content)
    write_split_data(output_prefix, train_data, dev_data, test_data)

if __name__ == "__main__":
    input_file = "data/E-NER-Dataset-main/all_tag_changed.txt"
    output_prefix = "data/E-NER-Dataset-main/ENER"
    main(input_file, output_prefix)