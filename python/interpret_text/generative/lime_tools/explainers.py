import torch
import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge, lars_path

from .text_utils import *
from .perturbation_tools import RemovalPerturbation


def run_klasso(featurization, loglikelihoods, weights,
               max_features=3, model_regressor=None, random_state=1):
    """
    Run the lime-style feature attribution with K-Lasso + locally-weighted linear models.
    featurization: Representations of the perturbed inputs.
    loglikelihoods: Log-likelihood of the output we are trying to explain.
    weights: The weight of each perturbed instance. More perturbed implies lower weight.
    """
    weighted_featurization = (
        (featurization - np.average(featurization, axis=0, weights=weights))
        * np.sqrt(weights[:, np.newaxis])
    )
    weighted_labels = (
        (loglikelihoods
         - np.average(
             loglikelihoods,
             weights=weights)
         ) * np.sqrt(weights))

    nonzero = range(weighted_featurization.shape[1])
    x_vector = weighted_featurization
    alphas, _, coefs = lars_path(x_vector,
                                 weighted_labels,
                                 method='lasso',
                                 verbose=False)

    for i in range(len(coefs.T) - 1, 0, -1):
        nonzero = coefs.T[i].nonzero()[0]
        if len(nonzero) <= max_features:
            break
    used_features = nonzero

    if model_regressor is None:
        # model_regressor = ElasticNet(alpha=0.25, l1_ratio=0.5, fit_intercept=True,
        #                        random_state=random_state)
        model_regressor = Ridge(alpha=1.0, fit_intercept=True,
                                random_state=random_state)

    model_regressor.fit(
        featurization[:, used_features], loglikelihoods, sample_weight=weights)

    explanation = list(sorted(zip(used_features,
                                  model_regressor.coef_),
                              key=lambda x: np.abs(x[1]),
                              reverse=True))
    return explanation


class LocalExplanations(ABC):
    def __init__(self, perturbation_model="removal",
                 partition_fn="sentences",
                 partition_kwargs={},
                 progress_bar=None):
        """
        perturbation_model: What perturbation strategy to use, e.g. removal or infilling
        partition_fn: How to partition the input
        partition_kwargs: Extra kwargs to the partition function, e.g. n to specify ngrams
        See the partition function below for the potential args
        """
        self.partition_fn = partition_fn
        self.partition_kwargs = partition_kwargs
        # Dry test
        self.partition("Dummy")
        if perturbation_model == "removal":
            self.perturbation_model = RemovalPerturbation()
        else:
            raise NotImplementedError()
        if progress_bar is None:
            def progress_bar(x):
                return x
        self.progress_bar = progress_bar

    def partition(self, text, *args, **kwargs):
        if self.partition_fn == "ngrams":
            assert "n" in self.partition_kwargs, "ngrams should have 'n' in the partition_kwargs"
            parts = extract_non_overlapping_ngrams(
                text, self.partition_kwargs["n"])

        elif self.partition_fn == "sentences":
            parts = split_into_sentences(text)

        elif self.partition_fn == "n_sentences":
            assert "n" in self.partition_kwargs, "n_sentences should have 'n' in the partition_kwargs"
            parts = split_k_sentences(text, self.partition_kwargs["n"])

        elif self.partition_fn == "dep_parse":
            parts = split_dep_parse(text)

        else:
            raise NotImplementedError(
                f"Partition function {self.partition_fn} not implemented.")

        return parts

    @abstractmethod
    def attribution(self, model, prompt, completion, *args, **kwargs):
        pass


class LocalExplanationLikelihood(LocalExplanations):
    def __init__(self,
                 perturbation_model="removal",
                 partition_fn="sentences",
                 partition_kwargs={},
                 progress_bar=None):
        super().__init__(
            perturbation_model,
            partition_fn,
            partition_kwargs,
            progress_bar)

    def attribution(self,
                    model_wrapped,
                    prompt,
                    completion,
                    max_parts_to_perturb=2,
                    n_samples=100,
                    max_features=3,
                    parts=None,
                    parts_to_replace=None,
                    prompt_fn=None,
                    *args, **kwargs):
        """
        Vanilla LIME

        model_wrapped: should contain get_conditional_loglikelihood_batch(texts, completion_offset)
        prompt: prompt to do the attribution with
        completion: completion to do the attribution with
        max_parts_to_perturb: maximum parts to perturb for a single instance
        n_samples: number of total perturbations
        parts: all of the parts, provide it if these are predetermined
        parts_to_replace: if not None, specifies the parts to perturb. if not provided, perturbs all parts.

        This will output the final attribution scores along with all the parts
        """
        prompt_fn = (lambda x: x) if prompt_fn is None else prompt_fn

        if parts is None:
            # Create partitions only if they are not fed from outside. This
            # requires a specified partition_fn
            assert self.partition_fn is not None
            parts = self.partition(prompt)
            parts_to_replace = np.arange(len(parts))

        perturbed_texts, indices = self.perturbation_model(parts,
                                                           parts_to_replace=parts_to_replace,
                                                           max_parts_to_perturb=max_parts_to_perturb,
                                                           n_samples=n_samples)
        # Add the original prompt
        perturbed_texts.append(prompt)
        indices.append([])
        perturbed_prompts = [prompt_fn(txt) for txt in perturbed_texts]
        try:  # TODO : Fix this hack for tiktoken tokenizers
            completion_len = len(
                model_wrapped.tokenizer(completion)["input_ids"])
        except BaseException:
            completion_len = len(model_wrapped.tokenizer.encode(completion))
        prompts = [t + completion for t in perturbed_prompts]
        completion_offset = torch.tensor(
            [completion_len for i in range(len(prompts))])
        perplexities, loglikelihoods = model_wrapped.get_conditional_loglikelihood_batch(
            texts=prompts, completion_offset=completion_offset, progress_bar=self.progress_bar)

        weights = np.array([len(idx) for idx in indices])
        # Featurization is simple 1-0.
        featurization = np.ones((loglikelihoods.shape[0], len(parts)))
        for i, idx in enumerate(indices):
            featurization[i, idx] = 0

        attribution_scores = run_klasso(
            featurization,
            loglikelihoods,
            weights,
            max_features=max_features)
        # Each item of attribution_scores contain (feat_idx, score) pairs.

        return attribution_scores, parts


class LocalExplanationHierarchical(LocalExplanations):
    def __init__(
            self,
            perturbation_model,
            perturbation_hierarchy,
            step_explainer,
            progress_bar=None):
        """
        partition_hierarchy should contain the hierarchy of partitionings to apply. example:
        [{{"partition_fn":  "n_sentences", "partition_kwargs": {"n": 3}},
        {"partition_fn": "sentences", "partition_kwargs": {}},
        {"partition_fn": "ngrams", "partition_kwargs": {"n": 3}}]
        """
        self.perturbation_hierarchy = perturbation_hierarchy
        self.perturbation_model = perturbation_model
        self.progress_bar = progress_bar
        self.step_explainer = step_explainer

    def attribution(self, model_wrapped,
                    prompt,
                    completion,
                    max_parts_to_perturb=3,
                    n_samples=100,
                    max_features=3,
                    prompt_fn=None,
                    intermediate_log_fn=None,
                    *args, **kwargs):
        """
        Hierarchical LIME

        model_wrapped: should contain get_conditional_loglikelihood_batch(texts, completion_offset)
        max_parts_to_perturb: maximum parts to perturb for a single instance
        n_samples: number of total perturbations
        prompt_fn: whether to format the prompt after perturbations

        This will output the final attribution scores for the parts along with the entire importance cache
        of the hierarchy
        """
        # TODO: Add a smarter strategy to pick number of features.
        # e.g. Can say each feature should be perturbed at least k times in
        # expectation.

        parts = None
        parts_to_replace = None
        importance_cache = {}
        for step_idx, perturbation_specs in enumerate(
                self.perturbation_hierarchy):
            max_feats_step = perturbation_specs["max_features"]
            perturbation_specs.pop("max_features")
            print(f"Hierarchy Step: {step_idx}")
            local_attribution = self.step_explainer(
                perturbation_model=self.perturbation_model,
                progress_bar=self.progress_bar,
                **perturbation_specs)
            if parts is not None:
                # Construct new parts
                temp_parts = []
                temp_parts_to_replace = []
                parts_idx = 0
                # Split each larger part into subparts with the next partition
                # in the hierarchy
                for i, part in enumerate(parts):
                    _temp_parts = local_attribution.partition(part)
                    temp_parts.extend(_temp_parts)
                    # Mark the new parts to replace
                    if i in parts_to_replace:
                        temp_parts_to_replace.extend(
                            [parts_idx + j for j in range(len(_temp_parts))])
                    parts_idx += len(_temp_parts)

                parts = temp_parts
                parts_to_replace = temp_parts_to_replace

            # Perform the attribution
            attribution_scores, parts = local_attribution.attribution(model_wrapped,
                                                                      prompt,
                                                                      completion,
                                                                      max_parts_to_perturb=max_parts_to_perturb,
                                                                      n_samples=n_samples,
                                                                      max_features=max_feats_step,
                                                                      parts=parts,
                                                                      parts_to_replace=parts_to_replace,
                                                                      prompt_fn=prompt_fn)

            # Take important features and only replace them in the future
            parts_to_replace = sorted([s[0] for s in attribution_scores])
            importance_cache[step_idx] = [
                parts, parts_to_replace, attribution_scores]
            if intermediate_log_fn is not None:
                step_idx = f"{step_idx+1}: {perturbation_specs}"
                intermediate_log_fn(
                    [parts, parts_to_replace, attribution_scores], step_idx)
        return importance_cache


class AttentionExplanationAllLayers:
    def __init__(self, agg_fn="sum"):
        agg_fns = {
            "sum": np.sum,
            "mean": np.mean,
            "max": np.max,
            "norm": np.linalg.norm,
        }
        self.agg_fn = agg_fns[agg_fn]

    def attribution(self,
                    model_wrapped,
                    prompt,
                    completion,
                    prompt_fn=None,
                    intermediate_log_fn=None,
                    *args, **kwargs):
        prompt_fn = (lambda x: x) if prompt_fn is None else prompt_fn
        full_prompt = prompt_fn(prompt) + completion
        completion_len = len(model_wrapped.tokenizer.encode(completion)[2:])
        prompt_tokens = model_wrapped.tokenizer.encode(prompt_fn(prompt))
        parts = [model_wrapped.tokenizer.decode([t]) for t in prompt_tokens]
        with torch.no_grad():
            inputs = model_wrapped.tokenizer(
                full_prompt, return_tensors="pt").to(
                model_wrapped.device)
            activations = model_wrapped.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_attentions=True)
            atts = activations["attentions"]
            atts = [att.float().cpu().numpy() for att in atts]
            atts = [self.agg_fn(att, axis=1) for att in atts]

        importance_cache = {}
        parts = parts[1:]
        for i, att in enumerate(atts):
            att = att[:, -completion_len:, :-completion_len]
            att = self.agg_fn(att, axis=-2).flatten()
            att = att[1:]
            attribution_scores = [(p, a)
                                  for p, a in zip(range(len(parts)), att)]
            if intermediate_log_fn is not None:
                intermediate_log_fn([parts, None, attribution_scores], i)
            importance_cache[i] = [parts, None, attribution_scores]
        return importance_cache


class LocalExplanationSentenceEmbedder(LocalExplanations):
    def __init__(self,
                 sentence_embedder,
                 perturbation_model="removal",
                 partition_fn="ngrams",
                 partition_kwargs={"n": 3},
                 progress_bar=None):
        super().__init__(
            perturbation_model,
            partition_fn,
            partition_kwargs,
            progress_bar)
        self.sentence_embedder = sentence_embedder
        self.embedder_instruction = "Represent the following model response:"

    def attribution(self,
                    model_wrapped,
                    prompt,
                    completion,
                    max_parts_to_perturb=2,
                    n_samples=100,
                    max_features=3,
                    parts=None,
                    parts_to_replace=None,
                    prompt_fn=None,
                    *args, **kwargs):
        """
        Vanilla LIME

        model_wrapped: should contain get_conditional_loglikelihood_batch(texts, completion_offset)
        prompt: prompt to do the attribution with
        completion: completion to do the attribution with
        max_parts_to_perturb: maximum parts to perturb for a single instance
        n_samples: number of total perturbations
        parts: all of the parts, provide it if these are predetermined
        parts_to_replace: if not None, specifies the parts to perturb. if not provided, perturbs all parts.

        This will output the final attribution scores along with all the parts
        """
        prompt_fn = (lambda x: x) if prompt_fn is None else prompt_fn

        if parts is None:
            # Create partitions only if they are not fed from outside. This
            # requires a specified partition_fn
            assert self.partition_fn is not None
            parts = self.partition(prompt)
            parts_to_replace = np.arange(len(parts))

        perturbed_texts, indices = self.perturbation_model(parts,
                                                           parts_to_replace=parts_to_replace,
                                                           max_parts_to_perturb=max_parts_to_perturb,
                                                           n_samples=n_samples)
        # Add the original prompt
        perturbed_texts.append(prompt)
        indices.append([])
        perturbed_prompts = [prompt_fn(txt) for txt in perturbed_texts]
        completion_tokens = len(model_wrapped.tokenizer.encode(completion))
        with torch.no_grad():
            new_embeddings = []
            for p in self.progress_bar(perturbed_prompts):
                new_output = model_wrapped.sample(
                    texts=[p], temperature=0, max_new_tokens=completion_tokens)[0]
                new_embeddings.append(self.sentence_embedder.encode(
                    [[self.embedder_instruction, new_output]]))
            print("New outputs generated")

            original_emb = self.sentence_embedder.encode(
                [[self.embedder_instruction, completion]])

        similarities = new_embeddings @ original_emb.T
        similarities = similarities.flatten()

        weights = np.array([len(idx) for idx in indices])
        # Featurization is simple 1-0.
        featurization = np.ones((similarities.shape[0], len(parts)))
        for i, idx in enumerate(indices):
            featurization[i, idx] = 0

        attribution_scores = run_klasso(
            featurization,
            similarities,
            weights,
            max_features=max_features)
        # Each item of attribution_scores contain (feat_idx, score) pairs.

        return attribution_scores, parts
