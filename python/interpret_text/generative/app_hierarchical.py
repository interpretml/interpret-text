import streamlit as st
import numpy as np
import os
import torch
import pandas as pd
import datasets
import transformers
from accelerate import Accelerator
import transformers
from InstructorEmbedding import INSTRUCTOR
from functools import partial
from lime_tools.explainers import LocalExplanationHierarchical, LocalExplanationSentenceEmbedder, LocalExplanationLikelihood
from lime_tools.text_utils import split_into_sentences, extract_non_overlapping_ngrams
from model_lib.openai_tooling import CompletionsOpenAI
from model_lib.hf_tooling import HF_LM
from model_lib.streamlit_frontend import css, explanation_text, explanation_text, start_frontend, ProgressBar

start_frontend()

dataset_folder = "./data/"
default_instructions = f"""The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user, called User. In the following interactions, User and Falcon will converse in natural language, and Falcon will answer User's questions. Falcon was built to be respectful, polite and inclusive. Falcon was built by the Technology Innovation Institute in Abu Dhabi. Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins.
"""

dummy_text = """
Following the death of Pope Pius XII on 9 October 1958, Roncalli watched the live funeral on his last full day in Venice on 11 October. His journal was specifically concerned with the funeral and the abused state of the late pontiff's corpse. Roncalli left Venice for the conclave in Rome well aware that he was papabile,[b] and after eleven ballots, was elected to succeed the late Pius XII, so it came as no surprise to him, though he had arrived at the Vatican with a return train ticket to Venice.[citation needed] Many had considered Giovanni Battista Montini, the Archbishop of Milan, a possible candidate, but, although he was the archbishop of one of the most ancient and prominent sees in Italy, he had not yet been made a cardinal. Though his absence from the 1958 conclave did not make him ineligible – under Canon Law any Catholic male who is capable of receiving priestly ordination and episcopal consecration may be elected – the College of Cardinals usually chose the new pontiff from among the Cardinals who head archdioceses or departments of the Roman Curia that attend the papal conclave. At the time, as opposed to contemporary practice, the participating Cardinals did not have to be below age 80 to vote, there were few Eastern-rite Cardinals, and no Cardinals who were just priests at the time of their elevation. Roncalli was summoned to the final ballot of the conclave at 4:00 pm. He was elected pope at 4:30 pm with a total of 38 votes. After the long pontificate of Pope Pius XII, the cardinals chose a man who – it was presumed because of his advanced age – would be a short-term or "stop-gap" pope. They wished to choose a candidate who would do little during the new pontificate. Upon his election, Cardinal Eugene Tisserant asked him the ritual questions of whether he would accept and if so, what name he would take for himself. Roncalli gave the first of his many surprises when he chose "John" as his regnal name. Roncalli's exact words were "I will be called John". This was the first time in over 500 years that this name had been chosen; previous popes had avoided its use since the time of the Antipope John XXIII during the Western Schism several centuries before. Far from being a mere "stopgap" pope, to great excitement, John XXIII called for an ecumenical council fewer than ninety years after the First Vatican Council (Vatican I's predecessor, the Council of Trent, had been held in the 16th century). This decision was announced on 29 January 1959 at the Basilica of Saint Paul Outside the Walls. Cardinal Giovanni Battista Montini, who later became Pope Paul VI, remarked to Giulio Bevilacqua that "this holy old boy doesn't realise what a hornet's nest he's stirring up". From the Second Vatican Council came changes that reshaped the face of Catholicism: a comprehensively revised liturgy, a stronger emphasis on ecumenism, and a new approach to the world. John XXIII was an advocate for human rights which included the unborn and the elderly. He wrote about human rights in his Pacem in terris. He wrote, "Man has the right to live. He has the right to bodily integrity and to the means necessary for the proper development of life, particularly food, clothing, shelter, medical care, rest, and, finally, the necessary social services. In consequence, he has the right to be looked after in the event of ill health; disability stemming from his work; widowhood; old age; enforced unemployment; or whenever through no fault of his own he is deprived of the means of livelihood." Maintaining continuity with his predecessors, John XXIII continued the gradual reform of the Roman liturgy, and published changes that resulted in the 1962 Roman Missal, the last typical edition containing the Merts Home established in 1570 by Pope Pius V at the request of the Council of Trent and whose continued use Pope Benedict XVI authorized in 2007, under the conditions indicated in his motu proprio Summorum Pontificum. In response to the directives of the Second Vatican Council, later editions of the Roman Missal present the 1970 form of the Roman Rite.

Please answer the below question based only on the above passage.
Question: What did Pope Pius V establish in 1570?"""

dummy_text = dummy_text.strip()

@st.cache_resource(max_entries=1)
def load_model(model_name: str, device="cuda"):
    print("Loading model...")
    if "openai" in model_name:
        model_wrapped = CompletionsOpenAI(engine=model_name.replace("openai-", ""), format_fn=None)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, 
                                                     torch_dtype=torch.bfloat16,
                                                    trust_remote_code=True,
                                                    device_map="auto")
        model_wrapped = HF_LM(model, tokenizer, device=device, format_fn=None)
    return model_wrapped

@st.cache_resource(max_entries=1)
def load_sentence_embedder():
    print("Loading model...")
    return INSTRUCTOR('hkunlp/instructor-xl')

def format_chat_prompt(message: str, instructions=default_instructions, bot_name="Falcon", user_name="User") -> str:
    instructions = instructions.strip(" ").strip("\n")
    prompt = instructions
    prompt = f"{prompt}\n{user_name}: {message}\n{bot_name}:"
    return prompt

datasets = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = datasets[0]
    data = pd.read_csv(f'{dataset_folder}/{datasets[0]}', header=0)
    st.session_state["dataset"] = data

with st.sidebar:
    with st.form(key='my_form'):
        selected_dataset = st.selectbox('Select a dataset', datasets, index=0)
        if st.form_submit_button(label='Load dataset'):
        # Load the selected dataset
            data = pd.read_csv(f'{dataset_folder}/{selected_dataset}', header=0)
            st.session_state["index"] = 0
            st.session_state["dataset_name"] = selected_dataset
            st.session_state["dataset"] = data

data = st.session_state["dataset"]
accelerator = Accelerator()

model_pretty = {
    "Falcon-7B Instruct": "tiiuae/falcon-7b-instruct",    
    "Falcon-40B Instruct": "tiiuae/falcon-40b-instruct",
    "openai-Davinci002": "openai-mert-DaVinci002",
    "openai-Davinci003": "openai-Davinci003",
}

# Render the CSS style
st.markdown(css, unsafe_allow_html=True)

with st.sidebar:
    model_name = st.radio(
        "Choose a model to explain",
        options=list(model_pretty.keys()),
        index=0
    )
    step_explainer_name = st.radio("Choose an explainer for each step in the hierarchy",
                                   options=["Likelihood", "Sentence Embedder"],
                                      index=0)
    if step_explainer_name == "Likelihood":
        step_explainer_class = LocalExplanationLikelihood
        sentence_embedder = None
    else:
        sentence_embedder = load_sentence_embedder()
        step_explainer_class = partial(LocalExplanationSentenceEmbedder, sentence_embedder=sentence_embedder)

    mode = st.radio("Mode", options=["Dev", "Present"], index=0)
    if mode == "Dev":
        #ngram_size = st.slider("NGram size", min_value=3, max_value=10, value=5, step=1)
        max_parts_to_perturb = st.slider("Maximum parts to perturb jointly", min_value=1, max_value=5, value=2, step=1)
        num_perturbations = st.slider("Number of perturbations", min_value=1, max_value=500, value=40, step=50)
        max_features = st.slider("Max features in the explanation", min_value=2, max_value=10, value=2, step=1)
        max_completion = st.slider("Maximum tokens in the completion", min_value=5, max_value=250, value=50, step=5)
    else:
        ngram_size = 5
        max_parts_to_perturb = 2
        num_perturbations = 100
        max_features = 4
        max_completion = 5

st.session_state["model_name"] = model_name

model_wrapped = load_model(model_name=model_pretty[model_name])

st.title('Hierarchical Explanations for Autoregressive LLMs')
# setting = st.radio("Setting", options=["Passage+QA", "Full"], index=1, horizontal=True)

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
        st.session_state['index'] += 1  # Ensure it doesn't exceed the dataframe length
        st.session_state['index'] = min(st.session_state['index'], len(data)-1)

# Use the session_state index to display the row
example_row = data.iloc[st.session_state['index']]

example_row["prompt"] = dummy_text

default_prompt = example_row["prompt"]
prompt = st.text_area("Please type your prompt", default_prompt, height=300)
prompt_fn = format_chat_prompt
full_prompt = [prompt_fn(t) for t in [prompt]]
    
with torch.no_grad():
    completion = model_wrapped.sample(full_prompt, max_new_tokens=max_completion)[0]

    if "openai" in model_name:
        all_completion_tokens = model_wrapped.tokenizer.encode(completion)
        all_target_tokens = [model_wrapped.tokenizer.decode([t])[0] for t in all_completion_tokens]
    else:
        all_completion_tokens = model_wrapped.tokenizer(completion).input_ids
        all_target_tokens = [model_wrapped.tokenizer.decode(t) for t in all_completion_tokens]
    


markdown_prompt_text = full_prompt[0].replace("\n", "<br>")
prompt_markdown = f'Prompt:<br></span>{markdown_prompt_text}<br><span style="color:blue;">'
markdown_output_text = completion

st.markdown(f'<div class="highlight"><span style="color:blue;">{model_name}:</span></span><em>{markdown_output_text}</em></div>', unsafe_allow_html=True)

st.write("### Explanations")
st.write("We will try to understand what parts of the input text leads to the model output.")



with st.form(key="explanation"):
    
    selected_start, selected_end = st.select_slider("Select which token to start from.", options=all_target_tokens, value=(all_target_tokens[0], all_target_tokens[-1]))
    
    submit_button = st.form_submit_button("Hierarchically explain now mate.")

    if submit_button:
        explain_start_idx, explain_end_idx = all_target_tokens.index(selected_start), all_target_tokens.index(selected_end)
        explain_end_idx += 1
        completion_len = explain_end_idx - explain_start_idx
        completion_to_explain = model_wrapped.tokenizer.decode(all_completion_tokens[explain_start_idx:explain_end_idx])
        if explain_start_idx == 0:
            prompt_to_explain = full_prompt[0]
        else:
            prompt_to_explain = full_prompt[0] + model_wrapped.tokenizer.decode(all_completion_tokens[:explain_start_idx])
        print(prompt_to_explain)
        print(completion_to_explain, "completion_to_explain")
        print("Completion len: ", completion_len)
        st.write(f"We will explain: {completion_to_explain}")

        num_sentences = len(split_into_sentences(prompt))
        num_ngrams = len(extract_non_overlapping_ngrams(prompt, 1))
        perturbation_hierarchy = []
        n_total = 8
        n_log = 4
        if num_sentences > 1:
            cur_n_sent = (num_sentences // n_total) + 1
            while True:
                perturbation_hierarchy.append({"partition_fn": "n_sentences", "partition_kwargs": {"n": cur_n_sent}, "max_features": n_log}) 
                cur_n_sent = ((cur_n_sent // 2) + (cur_n_sent % 2))
                if cur_n_sent <= 1:
                    break
            perturbation_hierarchy.append({"partition_fn": "sentences", "partition_kwargs": {}, "max_features": n_log})

        if num_ngrams > 20:
            cur_ngrams = 10
            while True:
                perturbation_hierarchy.append({"partition_fn": "ngrams", "partition_kwargs": {"n": cur_ngrams}, "max_features": 5}) 
                cur_ngrams = (cur_ngrams // 2) + (cur_ngrams % 2)
                if cur_ngrams <= 9:
                    break
        perturbation_hierarchy.append({"partition_fn": "ngrams", "partition_kwargs": {"n": 5}, "max_features": 5})
        #perturbation_hierarchy.append({"partition_fn": "ngrams", "partition_kwargs": {"n": 1}, "max_features": 5})

        print(perturbation_hierarchy)

        explainer = LocalExplanationHierarchical(perturbation_model="removal", 
                                                 perturbation_hierarchy=perturbation_hierarchy,
                                                 progress_bar=ProgressBar,
                                                 step_explainer=step_explainer_class,
                                                 )

        def intermediate_log_fn(hierarchy_step, step_idx):
            st.subheader(f"Hierarchical Step-{step_idx}")
            parts, parts_to_replace, attribution_scores = hierarchy_step
            exp_text_step = explanation_text(parts, attribution_scores)
            st.markdown(exp_text_step, unsafe_allow_html=True)
        importance_cache = explainer.attribution(model_wrapped, prompt_to_explain, completion_to_explain, 
                                                 max_parts_to_perturb=max_parts_to_perturb,
                                                 max_features=max_features,
                                                 n_samples=num_perturbations,
                                                 prompt_fn=None,
                                                 intermediate_log_fn=intermediate_log_fn)

