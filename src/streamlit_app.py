import streamlit as st
import torch

#Page configuration
st.set_page_config(
    page_title='Simple Demo for NER',
    layout = 'wide',
    initial_sidebar_state='expanded'
)

#Title of the app
st.title('Simple Demo for NER')

#Input widgets
# st.sidebar.subheader('Input features')
option = st.selectbox(
    'Choose the model for your NER task',
    ('General', 'Legal(full)', 'Legal(5%)', 'Legal(5% with transfer)')
)
if option == 'General':
    model = torch.load( "models/model_general_iobes_cpu")
elif option == 'Legal(full)':
    model = torch.load( "models/model_full_trained_on_ENER")
elif option == 'Legal(5%)':
    model = torch.load( "models/model_10k_train_ENER")
elif option == 'Legal(5% with transfer)':
    model = torch.load( "models/model_10k_train_ENER_transfer")
else:
    model = torch.load( "models/model_general_iobes_cpu")

st.write('Hello, you selected:', option)

input = st.text_input('Input text')
if input:
    st.write('The current input text is:', input)
    words = input.strip().split()
    st.write(words)

