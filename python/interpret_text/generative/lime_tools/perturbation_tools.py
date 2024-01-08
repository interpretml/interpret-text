import numpy as np
from abc import ABC, abstractmethod
from .text_utils import extract_non_overlapping_ngrams, split_into_sentences, split_dep_parse, split_k_sentences

class PerturbationModelBase(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def replace(self, parts, parts_to_replace=None, *args, **kwargs):
        """
        Different models should implement the replacement strategies here.
        """
        raise NotImplementedError()
        
    @abstractmethod
    def __call__(self, prompt, *args, **kwargs):
        """
        Different models should implement the replacement strategies here.
        """
        raise NotImplementedError()        
    
class RemovalPerturbation(PerturbationModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.perturb_strategy = "omit_text"
    
    def replace(self, parts, part_idx_to_replace, *args, **kwargs):
        """
        Simply omit text the LIME/SHAP way.
        """
        parts_modified = [p for i, p in enumerate(parts) if i not in part_idx_to_replace]
        return parts_modified
    
    def __call__(self, 
                 parts, 
                 max_parts_to_perturb=2, 
                 n_samples=50, 
                 parts_to_replace=None,
                 *args, **kwargs):
        """
        perturb the text, generation n_samples x perturbed texts.
        parts: list of parts to perturb
        parts_to_replace: indices of parts to replace
        max_parts_to_perturb: maximum number of chunks to replace
        n_samples: number of perturbed texts to generate
        """
        if parts_to_replace is None:
            parts_to_replace = np.arange(len(parts))
        
        assert max_parts_to_perturb <= (len(parts_to_replace)-1), "max_parts_to_perturb should be less than or equal to the number of parts to replace - 1"

        # Sample the number of chunks to replace
        sample_k = np.random.randint(1, max_parts_to_perturb+1, size=n_samples)

        indices = []
        perturbed_texts = []
        for k in sample_k:
            sampled_indices = np.random.choice(parts_to_replace, k, replace=False)
            perturbed_text = self.replace(parts, sampled_indices)
            # If consecutive ones are sampled, combine them into a single one. 
            perturbed_texts.append("".join(perturbed_text))
            indices.append(list(sorted(sampled_indices)))
        
        return perturbed_texts, indices
    

## TODO: Fix the thing below for the new perturbation model structure

class PerturbationModel:
    def __init__(self, mask_model, mask_tokenizer, 
                 partition_fn="split_chunks",
                 device="cuda",
                 num_processes=8):
        self.mask_model = mask_model
        self.mask_tokenizer = mask_tokenizer
        self.pattern = re.compile(r"<extra_id_\d+>")
        self.device = device
        self.partition_fn = split_chunks if partition_fn == "split_chunks" else extract_non_overlapping_ngrams
    
    def partition(self, *args, **kwargs):
        return self.partition_fn(*args, **kwargs)
        
    def tokenize_and_mask(self, text, ng=6, sample_k=2):
        text_chunks = self.partition(text, ng)
        chunk_idx = np.random.choice(len(text_chunks), min(sample_k, len(text_chunks)), replace=False)
        chunk_idx = sorted(chunk_idx)
        for i, idx in enumerate(chunk_idx):
            text_chunks[idx] = f"<extra_id_{i}>"
        return " ".join(text_chunks), list(chunk_idx)
    
    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
    
    def replace_masks(self, texts, batch_size=32, progress_bar=None, device="cuda"):
        print("Running perturbations")
        with torch.no_grad():
            n_expected = self.count_masks(texts)
            stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
            outputs = []
            for batch_texts in chunks(texts, batch_size, progress_bar=progress_bar, progress_text="Running perturbations"):
                tokens = self.mask_tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)
                outputs.extend(self.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=0.96, num_return_sequences=1, eos_token_id=stop_id))
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
    
    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    
        # return the text in between each matched mask token
        extracted_ids = [self.pattern.findall(x) for x in texts]
        extracted_fills = [self.pattern.split(x)[1:] for x in texts]
    
        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
        extracted_id_to_fill = [{idx: fill for idx, fill in zip(extracted_ids[i], extracted_fills[i])} for i, x in enumerate(texts)]
        return extracted_id_to_fill
    
    def apply_extracted_fills(self, masked_texts, extracted_id_to_fills):
        # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)
        
        for i, (text, n_fills) in enumerate(zip(tokens, n_expected)):
            for fill_idx in range(n_fills):
                fill_token = f"<extra_id_{fill_idx}>"
                if fill_token not in text:
                    continue
                if fill_token in extracted_id_to_fills[i]:
                    text[text.index(fill_token)] = extracted_id_to_fills[i][fill_token]
                else:
                    text[text.index(fill_token)] = ""
        
        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts
    
    def generate_k_perturbed(self, prompt, n_samples=100, ng=5, max_chunks=5, batch_size=16,
                             progress_bar=None, accelerator=None):
        """
        Generates the perturbed versions of the prompt. 
        prompt: Prompt to perturb.
        n_samples: How many perturbations to generate in one run
        ng: The size of the ngrams, ng=4 means operating over 4-grams.
        max_chunks: Maximum number of chunks to perturb in a single perturbation round.
        batch_size: Batch size of the perturbation runs.
        
        Returns: Perturbed texts and the indices of perturbed n-grams.
        """
        sample_k = np.random.randint(1, max_chunks+1, size=n_samples)
        indices = []
        masked_texts = []
        for k in sample_k:
            mt, ng_idx = self.tokenize_and_mask(prompt, ng=ng, sample_k=k)
            # If consecutive ones are sampled, combine them into a single one. 
            masked_texts.append(mt)
            indices.append(list(sorted(ng_idx)))
        raw_fills = self.replace_masks(masked_texts, batch_size=batch_size, progress_bar=progress_bar,
        device=accelerator.device)
        extracted_id_to_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_id_to_fills)
        return perturbed_texts, indices
