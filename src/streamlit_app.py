import streamlit as st
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import loader
from utils import *
from loader import *

#Page configuration
st.set_page_config(
    page_title='Simple Demo for Legal NER',
    layout = 'wide',
    initial_sidebar_state='expanded'
)

#Title of the app
st.title('Simple Demo for Legal NER')

#Input widgets
# st.sidebar.subheader('Input features')
option = st.selectbox(
    'Choose the model for your NER task',
    ('Legal(full)', 'Legal(5%)', 'Legal(5% with transfer)')
)
if option == 'Legal(full)':
    model_name = "models/model_full_trained_on_ENER"
    mapping_file = "models/mapping_model_full_trained_on_ENER.pkl"
elif option == 'Legal(5%)':
    model_name = "models/model_10k_train_ENER"
    mapping_file = "models/mapping_model_10k_train_ENER.pkl"
elif option == 'Legal(5% with transfer)':
    model_name = "models/model_10k_train_ENER_transfer"
    mapping_file = "models/mapping_model_10k_train_ENER_transfer.pkl"
else:
    model_name = "models/model_full_trained_on_ENER"
    mapping_file = "models/mapping_model_full_trained_on_ENER.pkl"

model = torch.load(model_name)

with open(mapping_file, "rb") as f:
    mappings = pickle.load(f)

word_to_id = mappings["word_to_id"]
tag_to_id = mappings["tag_to_id"]
char_to_id = mappings["char_to_id"]
parameters = mappings["parameters"]
word_embeds = mappings["word_embeds"]

id_to_tag = {id: tag for tag, id in tag_to_id.items()}

st.write('Hello, you selected:', option)

entity_type_to_color = {
    "LOC": "lightblue",
    "PER": "lightgreen",
    "MISC": "lightpink",
    "LEGISLATION/ACT": "lightcoral",
    "COURT": "lightsalmon",
    "ORG": "lightseagreen",
    "GOVERNMENT": "lightsteelblue",
}

input = st.text_input('Input text')
if input:
    prediction = []
    st.write('The current input text is:', input)
    str_words = input.strip().split()
    words = [word_to_id[w.lower() if w.lower() in word_to_id else '<UNK>']
                 for w in str_words]
    dwords = Variable(torch.LongTensor(words))

    chars2 = [[char_to_id[c if c in char_to_id else '<UNK>'] for c in w] for w in str_words]
    chars2_length = [len(c) for c in chars2]
    char_maxl = max(chars2_length)
    chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
    for i, c in enumerate(chars2):
        chars2_mask[i, :chars2_length[i]] = c

    chars2_mask = Variable(torch.LongTensor(chars2_mask))

    caps = [cap_feature(w) for w in str_words]
    dcaps = Variable(torch.LongTensor(caps))

    val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, {})
    predicted_id = out
    for (word, pred_id) in zip(str_words, predicted_id):
        # line = ' '.join([word, id_to_tag[pred_id]])
        line = (word, id_to_tag[pred_id][2:] if id_to_tag[pred_id] != 'O' else 'O')
        prediction.append(line)
    highlighted_text = ''

    current_entity_type = None
    for entity, entity_type in prediction:
        color = entity_type_to_color.get(entity_type, "white")
        if current_entity_type != entity_type:
            if current_entity_type is not None:
                highlighted_text += "</mark></span> "  # Close previous entity
            highlighted_text += f"<span style='display: inline-block; text-align: center;'><span style='font-size: 0.8em; line-height: 1; color: {color};'>{entity_type}</span><br/><mark style='background-color: {color};'>{entity} "
            current_entity_type = entity_type
        else:
            highlighted_text += f"{entity} "
    highlighted_text += "</mark></span>"  # Close the last entity

    st.subheader("Ouput")
    st.markdown(highlighted_text, unsafe_allow_html=True)


