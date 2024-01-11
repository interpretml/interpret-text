import streamlit as st
import pandas as pd
import os
from models import MODEL_LIST, get_model
from streamlit_frontend import start_frontend

dataset_folder = './data'

start_frontend()

# Assuming that the models are located in './m
# odels' directory


@st.cache_resource(max_entries=1)
def load_model(model_name):
    return get_model(model_name)


# Sidebar
st.sidebar.header('User Input Parameters')

# Model selection
selected_model = st.sidebar.selectbox(
    'Select a model', list(MODEL_LIST.keys()))

model = load_model(selected_model)
datasets = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]

selected_dataset = st.sidebar.selectbox('Select a dataset', datasets)
# Load the selected dataset
data = pd.read_csv(f'{dataset_folder}/{selected_dataset}', header=0)

st.sidebar.header('User Input Parameters')
temperature = st.sidebar.slider(
    'Temperature',
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01)
top_p = st.sidebar.slider(
    'Top P',
    min_value=0.0,
    max_value=1.0,
    value=0.96,
    step=0.01)
max_tokens = st.sidebar.slider(
    'Max Tokens',
    min_value=50,
    max_value=2000,
    value=200,
    step=50)

# A couple of hparams to collect
if 'index' not in st.session_state:
    st.session_state['index'] = 0
cols = st.columns(2)
with cols[0]:
    if st.button('Previous Row'):
        st.session_state['index'] -= 1  # Ensure it doesn't go below 0
        st.session_state['index'] = max(0, st.session_state['index'])

with cols[1]:
    if st.button('Next Row'):
        # Ensure it doesn't exceed the dataframe length
        st.session_state['index'] += 1
        st.session_state['index'] = min(
            st.session_state['index'], len(data) - 1)

st.markdown(
    "### TODO: fix the new row writing issue. should not react. fix the data.append issue.")
# Use the session_state index to display the row
example_row = data.iloc[st.session_state['index']]
st.dataframe(example_row, use_container_width=True)

st.write("Feel free to use the examples we have now, alternatively play around with it for yourself.")
st.write("Should you run into any interesting examples, "
         "please feel free to fill the form below and add it as a new row :)"
         )

prompt = st.text_area('Prompt',
                      value=data.iloc[st.session_state['index']]['prompt'],
                      height=50)
completion = model._sample_api_single(
    prompt,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p)

st.markdown(
    f'<div class="highlight"><span style="color:blue;">{selected_model}:</span></span><em>{completion}</em></div>',
    unsafe_allow_html=True)

with st.form("Add new row"):
    new_row = {col: [st.text_input(f"Enter new value for {col}")]
               for col in data.columns if col != 'prompt'}
    st.write("Prompt to add:")
    st.write(f"```{prompt}```")
    new_row["prompt"] = prompt

    if st.form_submit_button("Add new row"):
        # append new_row to the dataframe
        new_data = pd.DataFrame(new_row)
        data = pd.concat([data, new_data], ignore_index=True)
        data.to_csv(f'{dataset_folder}/{selected_dataset}', index=False)
        st.success('New row added successfully.')
