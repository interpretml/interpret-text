import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from scipy.special import log_softmax, logsumexp


def chunks(lst, n, progress_bar=None, progress_text="Batch computing"):
        for i in tqdm(range(0, len(lst), n)):
            if progress_bar is not None:
                progress_bar.progress(i/len(lst), text=progress_text)
            yield lst[i:i + n]

class HF_LM:
    def __init__(self, model, tokenizer, device="cuda", format_fn=None):
        """
        format_fn: A function to format the prompt. Defaults to identity.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.format_fn = format_fn if format_fn is not None else (lambda x: x)

    
    @torch.no_grad()
    def get_batch_loglikelihood(self, texts):
        """
        Compute the loglikelihood of the given set of texts.
        """
        perplexities = []
        total_likelihoods = []
        for text in texts:
            tokenized = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)
            labels = tokenized.input_ids
            outputs = self.model(input_ids=tokenized["input_ids"], 
                                 attention_mask=tokenized["attention_mask"], 
                                 labels=labels)
            logits = outputs.logits.cpu()
            labels = labels.cpu()

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").detach()
            ll_per_sample = -loss.view(shift_logits.shape[0], shift_logits.shape[1])
            nonpad_per_row = (shift_labels != -100).sum(dim=1)
            ll_mean = ll_per_sample.sum(dim=1)/nonpad_per_row

            ll_per_sample[(shift_labels == -100)] = 0
            ll_total = ll_per_sample.sum(dim=1)
            perplexities.append(ll_mean.cpu().numpy())
            total_likelihoods.append(ll_total.cpu().numpy())
        perplexities = np.concatenate(perplexities, axis=0)
        total_likelihoods = np.concatenate(total_likelihoods, axis=0)
        return perplexities, total_likelihoods 
    
    
    
    @torch.no_grad()
    def get_conditional_loglikelihood(self, texts: list, 
                                      completion_offset: torch.Tensor,
                                      tokens_to_explain: list = []):
        tokenized = self.tokenizer(
            texts, return_tensors="pt", truncation=True, return_token_type_ids=False, padding=True,
        ).to(self.device)
        labels = tokenized.input_ids
        outputs = self.model(**tokenized)
        logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)
        labels = labels.cpu()

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
        
        # Ignore the tokens prior to the completion offset. 
        mask = (torch.arange(shift_labels.size(1)).unsqueeze(0) < (shift_labels.shape[1]-completion_offset.unsqueeze(1)))
        # Choose which tokens to mask from LL computation.
        shift_labels[(mask | (shift_labels == self.tokenizer.pad_token_id))] = -100
        
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").detach()
        ll_per_sample = -loss.view(shift_logits.shape[0], shift_logits.shape[1])
        #nonpad_per_row = (shift_labels != -100).sum(dim=1)
        #ll_mean = ll_per_sample.sum(dim=1)/nonpad_per_row
        
        ll_per_sample[(shift_labels == -100)] = 0
        ll_total = ll_per_sample.sum(dim=1)
        torch.cuda.empty_cache()
        return ll_per_sample.float().cpu().numpy(), ll_total.float().cpu().numpy()
    
    @torch.no_grad()
    def get_conditional_loglikelihood_batch(self, texts: list,
                                            completion_offset: torch.Tensor,
                                            tokens_to_explain: list = [],
                                            progress_bar=None):
        """Padding messes with the likelihood computation. Will do 1 by 1 for now."""
        if progress_bar is None:
            progress_bar = tqdm
        ll_mean_all = []
        ll_total_all = []
        for i in progress_bar(range(len(texts))):
            llm, llt = self.get_conditional_loglikelihood(texts[i:i+1], completion_offset[i:i+1])
            ll_mean_all.append(llm)
            ll_total_all.append(llt)
        return ll_mean_all, np.concatenate(ll_total_all, axis=0)
    
    @torch.no_grad()
    def generate(self, texts, *args, **kwargs):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        generated = self.model.generate(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"], 
                                        pad_token_id=self.tokenizer.pad_token_id, *args, **kwargs)
        torch.cuda.empty_cache()
        return generated

    @torch.no_grad()
    def _sample_api_single(self, text, *args, **kwargs):
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs["max_tokens"]
            kwargs.pop("max_tokens")
        generated = self.generate([text], do_sample=False,
                     output_scores=True, return_dict_in_generate=True, *args, **kwargs)
        decoded = self.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0]
        target_completion = decoded.replace(text, "")
        return target_completion
    
    @torch.no_grad()
    def sample(self, texts, *args, **kwargs):
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs["max_tokens"]
            kwargs.pop("max_tokens")
        generated = self.generate(texts, do_sample=False, num_beams=1,
                     output_scores=True, return_dict_in_generate=True, *args, **kwargs)
        decoded = self.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
        target_completions = [d.replace(t, "") for d, t in zip(decoded, texts)]
        return target_completions       
    
    def eval_classification(self, dataset, n_debug=None):
        """
        Expects a dataset instance where returned items have the following attributes:
        - prompt: The prompt to evaluate on.
        - classes: A list of strings, each string is a class.
        - answer_index: The index of the correct class.

        n_debug: Number of examples to evaluate on for debugging purposes.
        """
        
        if n_debug is None:
            n_debug = len(dataset)
        examples = [item for item in dataset][:n_debug]

        prompts = [
            self.format_fn(example.prompt) + class_seq
            for example in examples
            for class_seq in example.classes
        ]
        target_token_lengths = [len(self.tokenizer(class_seq)["input_ids"]) for example in examples for class_seq in example.classes]

        _, ll = self.get_conditional_loglikelihood_batch(prompts, completion_offset=torch.tensor(target_token_lengths))
        pred_logprobs = ll.reshape((len(examples), -1))
        total_logprob = logsumexp(pred_logprobs, axis=1)
        normalised_logprobs = log_softmax(pred_logprobs, axis=1)
        pred_classes = np.argmax(pred_logprobs, axis=1)
        labels = np.array([e.answer_index for e in examples])
        losses = -normalised_logprobs[np.arange(labels.shape[0]), labels]
        metrics = {"Accuracy": (pred_classes == labels), "Loss": losses, "TotalProbSpanned": total_logprob}
        return metrics

