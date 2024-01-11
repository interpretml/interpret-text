import os
import tiktoken
import openai
from tqdm import tqdm
import time
import torch
import numpy as np
from scipy.special import logsumexp, log_softmax

DEFAULT_SYSTEM_PROMPT = "You are an AI assistant that helps people find information."

class ChatOpenAI:
    def __init__(self, engine="GPT35", system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.engine = engine
        pass

    def _sample_api_single(self, prompt, temperature=0, max_tokens=8000,
                           top_p=0.99):
        response = openai.ChatCompletion.create(
          engine=self.engine,
          messages = [{"role":"system","content":self.system_prompt},
                      {"role":"user","content":prompt}],
          temperature=temperature,
          max_tokens=max_tokens,
          top_p=top_p,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None)
        return response["choices"][0]["message"]["content"]
    


class CompletionsOpenAI:
    def __init__(self, engine="mert-DaVinci002", format_fn=None, encoding="p50k_base", api_settings=None):
        if api_settings is not None:
            openai.api_type = api_settings.get("api_type")
            openai.api_base = api_settings.get("api_base")
            openai.api_version = api_settings.get("api_version")
            openai.api_key = api_settings.get("api_key")
        self.engine = engine
        self.tokenizer = tiktoken.get_encoding(encoding)
        self.format_fn = format_fn if format_fn is not None else (lambda x: x)

    def _sample_api_single(self, prompt, temperature=0, max_tokens=50,
                           top_p=0.99):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            echo=False)
        return response["choices"][0]["text"]
    
    def sample(self, prompts, temperature=0, max_new_tokens=50, top_p=0.99):
        completions = []
        for p in prompts:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=p,
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                echo=False)
            completions.append(response["choices"][0]["text"])
        return completions

    def get_conditional_loglikelihood(self,
                                      texts: list, 
                                      completion_offset: int,
                                      tokens_to_explain: list = []):
        prompt = texts[0]
        retry = 0
        # TODO: make the below logic better.
        # Handle content filtering
        while True:
            retry += 1
            try:
                response = openai.Completion.create(
                      engine=self.engine,
                      prompt=prompt,
                      temperature=0,
                      max_tokens=0,
                      top_p=1.0,
                      frequency_penalty=0,
                      presence_penalty=0,
                      stop=None,
                      logprobs=0,
                      echo=True)
                break
            except Exception as e:
                print(e, prompt, "sleep")
                if retry >=10:
                    return -1, -1
                    break
                else:
                    time.sleep(1)
                    continue
        
        gen_tokens, token_logprobs = response["choices"][0]["logprobs"]["tokens"], response["choices"][0]["logprobs"]["token_logprobs"]
        target_logprobs = token_logprobs[-completion_offset:]
        target_tokens = gen_tokens[-completion_offset:]
        return -np.mean(target_logprobs), np.sum(target_logprobs)
        
    
    def get_conditional_loglikelihood_batch(self, 
                                            texts: list,
                                            completion_offset: torch.Tensor,
                                            tokens_to_explain: list = [],
                                            progress_bar=None):    
        if progress_bar is None:
            progress_bar = tqdm
        completion_offset = list(completion_offset)
        lls, pplxts = [], []
        
        for i in progress_bar(range(len(texts))):
            prompt = texts[i]
            offset = completion_offset[i]
            perplexity, loglikelihood = self.get_conditional_loglikelihood([prompt], offset)
            lls.append(loglikelihood)
            pplxts.append(perplexity)
    
        ll = np.array(lls)
        ppl = np.array(pplxts)
        return ppl, ll
    
    def eval_classification(self,
                               dataset,
                               n_debug):
        if n_debug is None:
            n_debug = len(dataset)
        examples = [item for item in dataset][:n_debug]

        prompts = [
            self.format_fn(example.prompt) + class_seq
            for example in examples
            for class_seq in example.classes
        ]
        target_token_lengths = [len(self.tokenizer.encode(class_seq)) for example in examples for class_seq in example.classes]

        _, ll = self.get_conditional_loglikelihood_batch(prompts, completion_offset=torch.tensor(target_token_lengths))
        pred_logprobs = ll.reshape((len(examples), -1))
        total_logprob = logsumexp(pred_logprobs, axis=1)
        normalised_logprobs = log_softmax(pred_logprobs, axis=1)
        pred_classes = np.argmax(pred_logprobs, axis=1)
        labels = np.array([e.answer_index for e in examples])
        losses = -normalised_logprobs[np.arange(labels.shape[0]), labels]
        metrics = {"Accuracy": (pred_classes == labels), "Loss": losses, "TotalProbSpanned": total_logprob}
        return metrics
