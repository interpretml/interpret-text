import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
from transformers import LlamaTokenizer, LlamaForCausalLM
from openai_tooling import CompletionsOpenAI, ChatOpenAI
from hf_tooling import HF_LM
from accelerate import Accelerator


MODEL_LIST = {
    "Davinci003": lambda: CompletionsOpenAI(engine="Davinci003"),
    "davinci002": lambda: CompletionsOpenAI(engine="davinci002"),
    "GPT3.5-Chat": lambda: ChatOpenAI(engine="GPT35"),
    "GPT3.5-Completions": lambda: CompletionsOpenAI(engine="GPT35"),
    "Falcon7B-Instruct": lambda: load_pytorch_model("tiiuae/falcon-7b-instruct"),
    "Falcon7B": lambda: load_pytorch_model("tiiuae/falcon-7b"),
    "Falcon40B-Instruct": lambda: load_pytorch_model("tiiuae/falcon-40b-instruct"),
    "Vicuna33B": lambda: load_pytorch_model("lmsys/vicuna-33b-v1.3"),
    "Guanaco": lambda: load_llama_based("JosephusCheung/Guanaco"),
    "Guanaco33B": lambda: load_llama_based("timdettmers/guanaco-33b-merged"),
    "Flan T5-XL": lambda: load_t5("google/flan-t5-xl"),
    "Flan T5-XXL": lambda: load_t5("google/flan-t5-xxl"),
    "MPT-30B": lambda: load_pytorch_model("mosaicml/mpt-30b"),
    "MPT-30B-Instruct": lambda: load_pytorch_model("mosaicml/mpt-30b-instruct"),
}

MODEL_LIST_VERIFIED = {
    "Falcon7B-Instruct": lambda: load_pytorch_model("tiiuae/falcon-7b-instruct"),
    "Falcon40B-Instruct": lambda: load_pytorch_model("tiiuae/falcon-40b-instruct"),
    "MPT-30B-Instruct": lambda: load_pytorch_model("mosaicml/mpt-30b-instruct"),
}


def load_pytorch_model(model_name, device="cuda"):
    print("Loading PyTorch model...")
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto")
    model = accelerator.prepare(model)
    model_wrapped = HF_LM(model, tokenizer, device=accelerator.device)
    return model_wrapped


def load_t5(model_name, device="cuda"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto")
    model = model.to(device)
    model_wrapped = HF_LM(model, tokenizer, device=device)
    return model_wrapped


def load_llama_based(model_name, device="cuda"):
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model_wrapped = HF_LM(model, tokenizer, device=device)
    return model_wrapped


def get_model(model_name):
    if model_name in MODEL_LIST:
        return MODEL_LIST[model_name]()
    else:
        raise ValueError(f"Model {model_name} not found.")
