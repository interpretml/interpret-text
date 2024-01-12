import streamlit as st
import numpy as np
import os
import torch
import pandas as pd
import json
from tqdm import tqdm
import transformers
from model_lib.hf_tooling import HF_LM
from easydict import EasyDict as edict
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt

from hooks import *
from model_lib.misc_utils import *
from model_lib.streamlit_frontend import css
from rare_knowledge.collection import *


@torch.no_grad()
def run_attention_monitor(prompts,
                          model_wrapped,
                          max_new_tokens=50):
    attention_layers = [
        (i, f"model.layers.{i}.self_attn") for i in range(
            model_wrapped.model.config.num_hidden_layers)]
    mlp_layers = [(i, f"model.layers.{i}.mlp") for i in range(
        model_wrapped.model.config.num_hidden_layers)]
    all_layers = attention_layers + mlp_layers
    all_layers.append([-1, "lm_head"])
    all_layers.append([-1, "model.norm"])

    all_data = []
    for prompt_info in tqdm(prompts):
        prompt = prompt_info["prompt"]
        prompt_len = len(model_wrapped.tokenizer.encode(prompt))
        with torch.no_grad():
            # Sample the model's completion
            completion = model_wrapped.sample(
                [prompt], max_new_tokens=max_new_tokens, temperature=0)[0]
        # set_requires_grad(True, model_wrapped.model)
        # Set retain_grad below to false.
        with torch.no_grad(), TraceDict(model_wrapped.model, [l_i[1] for l_i in all_layers], retain_grad=False) as ret:
            # Run the full text through the model.
            inputs = model_wrapped.tokenizer(
                prompt + completion,
                return_tensors="pt").to(
                model_wrapped.device)
            outs = model_wrapped.model(input_ids=inputs.input_ids,
                                       attention_mask=inputs.attention_mask,
                                       output_hidden_states=True,
                                       output_attentions=True)
            # Starting is just the token embeddings.
            hiddens = [h.detach().cpu().float().numpy()[0]
                       for h in outs["hidden_states"]][1:]

            # Backprop through the max logit of the first generation token.
            att_addition = []
            att_weights = []
            # att_weight_grads = []
            proj_contributions = []
            token_contribs = []
            pkvs = []
            mlps = []
            for (l, layername) in all_layers:
                if "self_attn" in layername:
                    o_proj = o_proj_matrices[l]
                    rep_layer = ret[layername].output
                    # a_i^l: will be T x d
                    att_addition.append(
                        rep_layer[0].detach().cpu().float().numpy()[0])
                    # A^{i,j}_{l} will be H x T_{C} x T_{G}
                    att_weights.append(
                        rep_layer[1].detach().cpu().float().numpy()[0])

                    # att_weight_grads.append(rep_layer[1].grad.detach().cpu().float().numpy()[0])
                    # past key value states
                    pkv = rep_layer[2][1][0]
                    # compute |A_{i, j}*h_{j}*W_{v}*W_{o}|
                    proj_contributions.append(
                        torch.einsum(
                            "HDd, HTd->HTD",
                            o_proj,
                            pkv).detach().cpu().float().numpy())
                    pkvs.append(pkv.detach().cpu().float().numpy())

                    token_contribs.append(
                        (att_weights[-1][:, prompt_len - 1, :, np.newaxis]
                         * proj_contributions[-1][:, :, :]).sum(axis=0)
                    )

                elif "mlp" in layername:
                    mlp_contribs = ret[layername].output
                    mlps.append(mlp_contribs.detach().cpu().float().numpy()[0])

                elif "lm_head" in layername:
                    logits = ret[layername].output.detach(
                    ).cpu().float().numpy()

                elif "model.norm" in layername:
                    postlayernorm = ret[layername].output.detach(
                    ).cpu().float().numpy()

            all_data.append({
                "attention_weights": np.stack(att_weights),
                # "att_weight_grads": np.stack(att_weight_grads),
                "attention_deltas": np.stack(att_addition),
                "pkvs": np.stack(pkvs),
                "token_contribs": np.stack(token_contribs),
                "hiddens": hiddens,
                "mlps": mlps,
                "proj_contribs": np.stack(proj_contributions),
                "completion": completion,
                "full_prompt": prompt + completion,
                **prompt_info
            })

    return all_data, (logits, postlayernorm)


def find_sub_list(sl, list, offset=0):
    sll = len(sl)
    for ind in (i for i, e in enumerate(list) if e == sl[0]):
        if ind < offset:
            continue
        if list[ind:ind + sll] == sl:
            return ind, ind + sll - 1


def cosine(A, B):
    return np.dot(A, B.T) / (np.linalg.norm(A) * np.linalg.norm(B))


def find_within_text(prompt, parts, tokenizer):
    """
    A function that identifies the indices of tokens of a part of the prompt.
    By default we use the first occurence.
    """
    prompt_tokens = model_wrapped.tokenizer.encode(prompt)
    part_tokens = [model_wrapped.tokenizer.encode(p)[2:] for p in parts]
    part_token_indices = [
        find_sub_list(
            pt, prompt_tokens) for pt in part_tokens]
    return part_token_indices


def load_data(dataset_name):
    if dataset_name == "schools":
        return load_schools()
    elif dataset_name == "songs":
        return load_songs()
    elif dataset_name == "football_teams":
        return load_football_teams()
    elif dataset_name == "basketball_players":
        return load_basketball_players()
    elif dataset_name == "words":
        return load_word_dataset()
    elif dataset_name == "trivia_qa":
        return load_trivia_qa_dataset()
    elif dataset_name == "hallucination_senator":
        senators = pd.read_csv(
            "./rare_knowledge/data/hallucination_senator.csv")
        senators = senators.to_dict(orient="records")
        return senators, None, None
    elif dataset_name == "counterfact":
        counterfact = json.load(open("./rare_knowledge/data/counterfact.json"))
        for item in counterfact:
            item["prompt"] = item["requested_rewrite"]["prompt"].format(
                item["requested_rewrite"]["subject"])
        return counterfact, None, None
    else:
        raise ValueError()


st.set_page_config(page_title="Model Internals Dashboard", layout="wide")

# Render the CSS style
st.markdown(css, unsafe_allow_html=True)


default_prompt = (
    'User: Is there a person 1) whose name starts with Z 2) who is a football '
    'player from Fenerbahce?\nAssistant: Yes, his name is'
)


@st.cache_resource(max_entries=1)
def load_model(model_name: str):
    print("Loading model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

    model = model.eval()
    model = model.to("cuda")
    model_wrapped = HF_LM(model, tokenizer, device="cuda")
    model_wrapped.o_proj_matrices = [
        torch.stack(
            p.weight.split(
                5120 // 40,
                dim=1)).detach() for nm,
        p in model_wrapped.model.named_modules() if "o_proj" in nm]
    model_wrapped.linear_head = model_wrapped.model.lm_head.weight.detach().cpu().float().numpy()
    return model_wrapped


model_pretty = {
    "Llama2-HF": "meta-llama/Llama-2-13b-hf",
    "Llama2-7B-HF": "meta-llama/Llama-2-7b-hf",
}

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "football_teams"
    data = load_data("football_teams")
    st.session_state["dataset"] = data[0]
    st.session_state["index"] = 0


with st.sidebar:
    model_name = st.radio(
        "Choose a model to explain",
        options=list(model_pretty.keys()),
        index=1
    )

    with st.form(key='my_form'):
        dataset_name = st.selectbox("Choose a dataset to browse", options=list(
            [l_i.split(".")[0] for l_i in os.listdir("./rare_knowledge/data")]), index=0)
        if st.form_submit_button(label='Load dataset'):
            # Load the selected dataset
            data, _, _ = load_data(dataset_name)
            st.session_state["index"] = 0
            st.session_state["dataset_name"] = dataset_name
            st.session_state["dataset"] = data

    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=10,
        max_value=50,
        value=20,
        step=10)


st.session_state["model_name"] = model_name

model_wrapped = load_model(model_name=model_pretty[model_name])
o_proj_matrices = model_wrapped.o_proj_matrices
linear_head = model_wrapped.linear_head

st.title('Model Internals Deepdive for Failure Detection')

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
            st.session_state['index'], len(
                st.session_state["dataset"]) - 1)

with st.form(key="dashboard"):
    print(st.session_state["dataset"][st.session_state["index"]])
    prompt = st.session_state["dataset"][st.session_state["index"]]["prompt"]
    prompt = st.text_area("Prompt", value=prompt)
    constraint_text = st.text_input(
        "Constraint Text (single constraint for now)", value="")
    constraint_to_search = " " + constraint_text.strip()
    submit_button = st.form_submit_button("Render the Dashboard.")

    if submit_button:
        prompt_infos = [{"prompt": prompt,
                         "fillers": [constraint_to_search],
                         "templates": [constraint_to_search]}]
        all_data, (logits, postlayernorm) = run_attention_monitor(
            prompt_infos, model_wrapped, max_new_tokens=max_new_tokens)
        print(all_data[0]["hiddens"][-1] - postlayernorm)
        st.write(f"Completion: {all_data[0]['completion']}")
        plt.close("all")
        sns.set_context("poster")
        sns.set_style("whitegrid")
        all_records = []
        toptoken_data = []
        flow_data = []
        for k in tqdm(range(1)):
            data = edict(all_data[k])
            # Sets of indices
            try:
                filler_indices = find_within_text(
                    data.prompt, data.fillers, model_wrapped.tokenizer)
                # print(filler_indices)
            except Exception as e:
                st.write(f"Could not find the constraint: {e}")
                filler_indices = []
            # Get the locations of the filler tokens and template tokens.

            completion_tokens = find_within_text(
                data.full_prompt, [
                    data.completion], model_wrapped.tokenizer)
            n_prompt = len(model_wrapped.tokenizer.encode(data.prompt))
            gs, ge = n_prompt - 1, data["attention_weights"].shape[-1] - 1

            full_prompt_tokens = model_wrapped.tokenizer.encode(
                data["full_prompt"])
            # print(model_wrapped.tokenizer.decode(full_prompt_tokens[gs+1:ge+1]), "full prompt")
            records = []
            first_token = model_wrapped.tokenizer.encode(data.completion)[
                2:][0]
            final_reps = data["hiddens"][-1]

            final_pred = np.argmax(logits[k][gs])
            st.write(
                f"final pred {model_wrapped.tokenizer.decode(final_pred)}")
            pred_vector = linear_head[final_pred] - linear_head.mean(axis=0)
            # pred_vector_normalized = pred_vector / np.linalg.norm(pred_vector)

            for l_i in range(data["attention_weights"].shape[0]):
                # H x T x T
                attn_weights = data["attention_weights"][l_i][:, gs:ge]
                pkvs = data["pkvs"][l_i]  # H x T_i x d
                proj_contribs = data["proj_contribs"][l_i]  # H x T x D
                mlps = data["mlps"][l_i]
                num_heads = attn_weights.shape[0]
                hidden_dims = mlps.shape[-1]
                if l_i > 0:
                    hidden_prev = data["hiddens"][l_i - 1]
                else:
                    hidden_prev = np.ones_like(data["hiddens"][0])
                token_contribs_per_head = np.linalg.norm(
                    (data["attention_weights"][l_i][:, gs, :, np.newaxis] * proj_contribs[:, :, :]), axis=-1)
                token_contribs = data["token_contribs"][l_i]
                contrib_norms = np.linalg.norm(token_contribs, axis=-1)
                att_alignments = cosine(
                    token_contribs, data["hiddens"][-1][gs])
                mlp_alignments = cosine(mlps, data["hiddens"][-1][gs])
                hidden_norm = np.linalg.norm(data["hiddens"][l_i][gs])

                if l_i > 0:
                    mlps_prev = data["mlps"][l_i - 1]
                    att_alignments_prev = cosine(
                        data["token_contribs"][l_i - 1], data["hiddens"][-1][gs])

                for t in range(3, n_prompt):
                    token_label = model_wrapped.tokenizer.decode(
                        full_prompt_tokens[t])
                    next_token_label = model_wrapped.tokenizer.decode(
                        full_prompt_tokens[t + 1])
                    if token_label in ["\n", "Assistant", "Ass"]:
                        break
                    metadata = {
                        "layer": l_i,
                        "token_idx": t,
                        "data_idx": k,
                        "label": token_label}
                    flow_data.append({
                        "value": contrib_norms[t] / hidden_norm,
                        "type": r"$||a_{i,T}^{\ell}||$",
                        "orientation": "to",
                        **metadata
                    })

                    flow_data.append({
                        "value": attn_weights[:, 0, t].max(),
                        "type": r"$\max_{h} A_{i,T}^{\ell, h}$",
                        "orientation": "to",
                        **metadata
                    })

                    # flow_data.append({
                    #    "value": att_alignments[t],
                    #    "type": r"Alignment $\langle a_{c,g}^{\ell}, x_{g}^{L} \rangle$",
                    #    "orientation": "to",
                    #    **metadata
                    # })

                token_contribs_prev = token_contribs

                hidden_l = data["hiddens"][l_i]

                # Apply rmsnorm
                if (l_i == (len(all_data[0]["hiddens"]) - 1)):
                    final_probs = softmax(
                        linear_head @ hidden_l[n_prompt - 1:n_prompt + 2].T, axis=0)
                else:
                    # If we want to do post-layernorm
                    # hidden_up = (model_wrapped.model.model.norm(
                    #         torch.tensor(hidden_l[n_prompt-1:n_prompt+2]).unsqueeze(dim=0).cuda().bfloat16()
                    #     ).float().cpu().detach().numpy()[0]
                    # )
                    # final_probs = softmax(linear_head @ hidden_up.T, axis=0)
                    final_probs = softmax(
                        linear_head @ hidden_l[n_prompt - 1:n_prompt + 2].T, axis=0)

                for i in range(final_probs.shape[-1]):
                    toptoken_idx = final_probs[:, i].argmax()
                    toptoken = model_wrapped.tokenizer.decode(toptoken_idx)
                    top_prob = final_probs[toptoken_idx, i]
                    toptoken_text = f"{toptoken}\n({top_prob:.2f})"
                    toptoken_data.append(
                        {
                            "layer": l_i,
                            "token_idx": gs,
                            "toptoken": toptoken_text,
                            "data_idx": k,
                            "prompt": data.prompt,
                            "fillers": " ".join(
                                data["fillers"]),
                            "gen_token": f"{i+1}:{model_wrapped.tokenizer.decode(full_prompt_tokens[n_prompt+i])}",
                        })

        df_all = pd.DataFrame(all_records)
        df_toptoken = pd.DataFrame(toptoken_data)
        df_toptoken = df_toptoken[df_toptoken.data_idx == 0]
        # Pivot the data to match the desired format
        df_pivot = df_toptoken[((df_toptoken.layer > 5))].pivot(
            index='layer', columns='gen_token', values='toptoken')
        df_pivot = df_pivot.transpose()  # Transpose the table
        fig_table, ax = plt.subplots(1, 1, figsize=(25, 10))

        # Hide axes
        ax.axis('off')

        # Create table and display
        table = ax.table(cellText=df_pivot.values,
                         colLabels=df_pivot.columns,
                         rowLabels=df_pivot.index,
                         cellLoc='center',
                         loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 3)

        def plot_attention_trace(result, savepdf=None, title=None):
            plt.close("all")
            from kondo import use_style
            use_style(palette="bmh")
            scores = result["scores"]
            labels = result["labels"][:scores.shape[0]]
            scores[scores <= scores.max() * 0.25] = 0
            sns.set_context("talk")
            fig, ax = plt.subplots(figsize=(5, 4.5))
            ax.invert_yaxis()
            ax.set_yticks([0.5 + i for i in range(len(scores))])
            ax.set_xticks([0.5 + i for i in range(0, scores.shape[1] - 1, 10)])
            ax.set_xticklabels(list(range(0, scores.shape[1] - 1, 10)))
            ax.set_yticklabels(labels, fontsize=12)

            ax.set_title("Attention from Tokens to the Generation")
            ax.set_xlabel(r"Layer ($\ell$)")
            if title is not None:
                ax.set_title(title)
            fig.tight_layout()
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            fig.savefig(savepdf)
            return fig

        st.subheader("Flow To Generation")

        flowdf = pd.DataFrame(flow_data)

        cols = st.columns(len(flowdf.type.unique()))
        for m_idx, metric in enumerate(flowdf.type.unique()):
            flowdf = pd.DataFrame(flow_data)
            flowdf = flowdf[flowdf.orientation == "to"]
            flowdf = flowdf[flowdf.type == metric]
            flowdf = flowdf[flowdf.token_idx > 0]

            for d in flowdf.data_idx.unique():
                print(all_data[d]["full_prompt"])
                sns.set_context("paper")
                df_sample = flowdf[flowdf.data_idx == d]
                df_pivot = df_sample.pivot(
                    index="token_idx", columns="layer", values="value")
                token_idx = df_pivot.shape[0]
                labels = [df_sample[df_sample.token_idx == t]["label"].unique(
                )[0] for t in df_sample.token_idx.unique() if t > 0]
                fig = plot_attention_trace(
                    {
                        "scores": df_pivot,
                        "labels": labels},
                    title=metric,
                    savepdf=f"./figures/tools_attention_{metric}.pdf")
                with cols[m_idx]:
                    st.pyplot(fig)

        st.pyplot(fig_table, use_container_width=True)
