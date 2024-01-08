## The below code is taken from https://github.com/naimenz/inverse-scaling-eval-pipeline

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import os
from typing import Union, cast, Sequence
from typing_extensions import Literal, get_args
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig  # type: ignore
from huggingface_hub import snapshot_download

from accelerate import Accelerator

from .isp_dataset import (
    ClassificationExample,
    Example,
    ExampleWithClasses,
    LogoddsExample,
    SequenceProbExample,
    NumericExample,
    TaskType,
)
from .isp_utils import BasicParser
from .isp_utils import APIParameters, BaseGPT3Model, OpenAIModel, call_api, OpenaAIModelList

#OPENAI_API_BASE_URL = "https://api.openai.com/v1/engines"
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# for checking how long the input is

ValidHFModel = Literal[
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "gpt-neo-125M",
    "gpt-neo-1.3B",
    "gpt-neo-2.7B",
    "gpt-j-6B",
    "opt-125m",
    "opt-350m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct"
]
valid_hf_models: tuple[ValidHFModel, ...] = get_args(ValidHFModel)

# NOTE: due to limitations of get_args with nested Literals, we can't directly call get_args on OpenAIModel
valid_gpt3_models = OpenaAIModelList
Device = Literal["cuda:0", "cpu"]


class Model(ABC):
    @abstractmethod
    def __call__(
        self, examples: list[Example], task_type: TaskType
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        raise NotImplementedError("Abstract method")

    @staticmethod
    def from_name(
        model_name: Union[ValidHFModel, OpenAIModel], device: Device
    ) -> Model:
        if model_name in valid_hf_models:
            model = HFModel(model_name, device)
        elif model_name in valid_gpt3_models:
            model = GPT3Model(model_name)
        else:
            raise ValueError(f"Unrecognised model '{model_name}'")
        return model


class HFModel(Model):
    def __init__(self, model_name: ValidHFModel, device: Device) -> None:
        self.device = device
        # have to download the opt models in advance since they're new
        # The OPT models have a start token that we need to remove in some places
        self.correction_for_start_token = 0
        accelerator = Accelerator()
        if model_name.startswith("opt-"):
            prefix = "facebook/"
            self.model = self._load_opt(prefix + model_name, device)
            self.correction_for_start_token = 1
        else:
            if model_name.startswith("gpt-neo") or model_name.startswith("gpt-j"):
                prefix = "EleutherAI/"
            else:
                prefix = ""
            torch.cuda.empty_cache()
            self.model = AutoModelForCausalLM.from_pretrained(prefix + model_name, 
                                                              max_length=1024,
                                                              trust_remote_code=True,
                                                              device_map="auto")
        self.model = accelerator.prepare(self.model)
        self.device = accelerator.device
        # apparently the OPT models need slightly different tokenizers
        # https://huggingface.co/docs/transformers/main/en/model_doc/opt#overview
        if model_name.startswith("opt-"):
            use_fast = False
        else:
            use_fast = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            prefix + model_name,
            use_fast=use_fast,
            model_max_length=1023,
        )

    def _load_opt(self, checkpoint: str, device: Device):
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
            max_length=1024,
        )
        return self.model

    def __call__(
        self, examples: list[Example], task_type: TaskType
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        # TODO: remove this restriction
        if len(examples) > 1:
            raise ValueError(
                f"Batch size of {len(examples)} not currently supported for HF models: please use 1"
            )
        with torch.no_grad():
            if task_type.startswith("classification"):
                classification_examples = cast("list[ClassificationExample]", examples)
                rv = self._evaluate_classification(
                    classification_examples, task_type=task_type
                )
            elif task_type == "numeric":
                numeric_examples = cast("list[NumericExample]", examples)
                rv = self._evaluate_numeric(numeric_examples)
            elif task_type == "sequence_prob":
                sequence_prob_examples = cast("list[SequenceProbExample]", examples)
                rv = self._evaluate_sequence_prob(sequence_prob_examples)
            elif task_type == "logodds":
                logodds_examples = cast("list[LogoddsExample]", examples)
                rv = self._evaluate_logodds(logodds_examples, take_absolute_value=False)
            elif task_type == "absolute_logodds":
                logodds_examples = cast("list[LogoddsExample]", examples)
                rv = self._evaluate_logodds(logodds_examples, take_absolute_value=True)
            else:
                raise ValueError(f"Unrecognised task type {task_type}")
            return rv

    def _evaluate_classification(
        self,
        examples: list[ClassificationExample],
        task_type: TaskType,
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        prompts = [
            example.prompt + class_seq
            for example in examples
            for class_seq in example.classes
        ]
        all_logits, all_tokens = self._get_logits_and_tokens(prompts)
        total_logprobs = []
        losses = []
        labels_correct = []
        labels_predicted = []
        prompt_start = 0
        for example in examples:
            n_classes = len(example.classes)
            class_logprobs = []
            for j in range(n_classes):
                class_index = prompt_start + j
                class_logits = all_logits[class_index]
                # the lengths of each class sequence in tokens
                class_sequence = example.classes[j]
                # NOTE: we subtract 1 if OPT because the first token is the start of the sequence
                target_token_length = (
                    len(self.tokenizer(class_sequence)["input_ids"])
                    - self.correction_for_start_token
                )
                # we only need the logits for the end sequence
                tokens = all_tokens[class_index]
                # we have to go back by one because we don't care about the logits for the predicted token
                sequence_logits = class_logits[-target_token_length - 1 : -1]
                sequence_tokens = tokens[-target_token_length:]
                # we take a log_softmax over all token logits for each position in the class sequence to
                #  get log probabilities, and then sum the logprobs for the tokens actually chosen
                logprobs = F.log_softmax(sequence_logits, dim=-1)
                class_logprob = sum(
                    [logprobs[i, token] for i, token in enumerate(sequence_tokens)]
                )
                class_logprobs.append(class_logprob.item())  # type: ignore (the sum is never empty so never just 0, always a tensor)

            total_logprob = torch.logsumexp(torch.tensor(class_logprobs), dim=-1).item()
            normalised_logprobs = F.log_softmax(torch.tensor(class_logprobs), dim=-1)
            loss = -normalised_logprobs[example.answer_index].item()
            label_correct = int(np.argmax(normalised_logprobs) == example.answer_index)
            total_logprobs.append(total_logprob)
            losses.append(loss)
            labels_correct.append(label_correct)

            label_predicted = example.classes[
                torch.tensor(class_logprobs).argmax(dim=-1).item()
            ]
            labels_predicted.append(label_predicted)

            prompt_start += n_classes
        return {
            "loss": losses,
            "correct": labels_correct,
            "predicted": labels_predicted,
            "total_logprob": total_logprobs,
        }

    def _get_logits_and_tokens(
        self, prompts: list[str]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        all_logits = []
        all_tokens = []
        for prompt in prompts:
            tokenized_inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, return_token_type_ids=False,
            ).to(self.device)
            outputs = self.model(**tokenized_inputs)
            logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)
            # need to remove batch dimension
            all_logits.append(torch.squeeze(logits))
            all_tokens.append(torch.squeeze(tokenized_inputs["input_ids"]))
        return all_logits, all_tokens

    def _evaluate_sequence_prob(
        self, examples: list[SequenceProbExample]
    ) -> dict[str, Sequence[float]]:
        # finding the target
        prompts = [example.prompt + example.completion for example in examples]
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True, return_token_type_ids=False
        ).to(self.device)

        target_sequences = [example.completion for example in examples]
        # NOTE: we have to apply the OPT token correction here too
        target_token_lengths = [
            len(self.tokenizer(word)["input_ids"]) - self.correction_for_start_token
            for word in target_sequences
        ]

        outputs = self.model(**tokenized_inputs)
        logits = outputs["logits"].detach().to(device="cpu", dtype=torch.float32)

        losses = []
        for i in range(len(examples)):
            # we only need the logits for the end sequence
            tokens = tokenized_inputs["input_ids"][i]
            # we have to go back by one because we don't care about the logits for the predicted token
            sequence_logits = logits[i, -target_token_lengths[i] - 1 : -1]
            sequence_tokens = tokens[-target_token_lengths[i] :]
            logprobs = -F.log_softmax(sequence_logits, dim=-1)
            loss = sum([logprobs[i, token] for i, token in enumerate(sequence_tokens)])
            losses.append(loss.item())  # type: ignore (the sum is never empty so never just 0, always a tensor)
        return {"loss": losses}

    def _evaluate_logodds(
        self,
        examples: list[LogoddsExample],
        take_absolute_value: bool = False,
    ) -> dict[str, Union[Sequence[float], Sequence[int]]]:
        """logodds is much like classification, except we need to compare across prompts so we just
        compute the log odds here"""
        prompts = [example.prompt for example in examples]
        other_prompts = [example.other_prompt for example in examples]
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True, return_token_type_ids=False
        ).to(self.device)
        other_tokenized_inputs = self.tokenizer(
            other_prompts, return_tensors="pt", truncation=True, return_token_type_ids=False
        ).to(self.device)
        outputs = self.model(**tokenized_inputs)
        other_outputs = self.model(**other_tokenized_inputs)
        # we only need the logits for the final (new) token
        # NOTE: this may need to change if we use batch size > 1 with padding
        logits = outputs["logits"][:, -1].detach().to(device="cpu", dtype=torch.float32)
        other_logits = (
            other_outputs["logits"][:, -1]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        logodds = self._logodds_from_logits(examples, logits)
        other_logodds = self._logodds_from_logits(examples, other_logits)

        logodds_differences = list(np.array(logodds) - np.array(other_logodds))  # type: ignore (np typing bad)
        answer_indices = [example.answer_index for example in examples]
        # flip the order (and hence the sign) if the answer is "no"
        # (unless we are taking absolute values)
        for i, answer_index in enumerate(answer_indices):
            if answer_index == 1:
                logodds_differences[i] *= -1
            if take_absolute_value:
                logodds_differences[i] = np.abs(logodds_differences[i])

        accuracies = self._accuracies_from_logits(examples, other_logits)
        total_logprob = list(
            torch.logsumexp(
                torch.stack(
                    (
                        torch.tensor(
                            self._total_logprobs_from_logits(examples, logits)
                        ),
                        torch.tensor(
                            self._total_logprobs_from_logits(examples, other_logits)
                        ),
                    )
                ),
                dim=0,
            )
        )
        return {
            "logodds_difference": logodds_differences,
            "correct": accuracies,
            "total_logprob": total_logprob,  # type: ignore (they should be floats)
        }

    def _evaluate_numeric(
        self, examples: list[NumericExample]
    ) -> dict[str, Sequence[float]]:
        prompts = [example.prompt for example in examples]
        tokenized_inputs = self.tokenizer(
            prompts, return_tensors="pt", truncation=True, return_token_type_ids=False
        ).to(self.device)
        parser = BasicParser()
        # NOTE: this may need to change if we use batch size > 1 with padding
        outputs = self.model.generate(
            **tokenized_inputs,
            do_sample=True,
            num_return_sequences=10,
            max_new_tokens=7,
            temperature=0.5,
            pad_token_id=50526,
        )
        full_completions = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        # strip out the prompt NOTE: again we're assuming the batch_size is 1
        untrimmed_completions = [
            fc[len(examples[0].prompt) :] for fc in full_completions
        ]
        # dropping anything after a new line
        completions = [comp.split("\n")[0] for comp in untrimmed_completions]
        floats = parser(completions)
        # for now, we'll just take the mean of valid outputs as the estimate
        valid_floats = [f for f in floats if f is not None]
        if len(valid_floats) > 0:
            estimate = sum(valid_floats) / len(valid_floats)
        else:
            raise ValueError("No valid numbers returned")
        return {"estimate": [estimate]}

    def _logodds_from_logits(
        self, examples: list[LogoddsExample], logits: torch.Tensor
    ) -> list[float]:
        """Given examples and logits for those examples,
        compute the binary log odds for each example"""
        logodds_list = []
        for i, example in enumerate(examples):
            relevant_logits = self._extract_relevant_logits(logits, example, i)
            logprobs = F.log_softmax(relevant_logits, dim=-1)
            # NOTE: assuming always binary
            if len(logprobs) != 2:
                raise ValueError(f"Expected len(logprobs) == 2, not {len(logprobs)}")
            logodds = logprobs[0] - logprobs[1]
            logodds_list.append(logodds.item())
        return logodds_list

    def _accuracies_from_logits(self, examples, logits) -> list[int]:
        """Given examples and logits for those examples,
        compute whether the predicted label is correct for each example"""
        labels_correct = []
        for i, example in enumerate(examples):
            relevant_logits = self._extract_relevant_logits(logits, example, i)
            label_correct = int(
                np.argmax(relevant_logits.cpu().detach().numpy())
                == example.answer_index
            )
            labels_correct.append(label_correct)
        return labels_correct

    def _extract_relevant_logits(
        self, logits: torch.Tensor, example: ExampleWithClasses, index: int
    ) -> torch.Tensor:
        example_logits = logits[index]
        # NOTE: we take the last element of the returned token list
        # this is because the tokenizer returns a 1-element list for GPT tokenizers
        # and a 2-element list with start token in the first position for OPT tokenizers
        class_tokens = [
            token[-1] for token in self.tokenizer(list(example.classes))["input_ids"]
        ]
        # log_softmax just subtracts a constant, so repeated applications change nothing
        # and there is no point in taking logprobs before focusing on the relevant indices
        relevant_logits = example_logits[class_tokens]
        return relevant_logits

    def _total_logprobs_from_logits(self, examples, logits) -> list[float]:
        """Given examples and logits for those examples,
        compute the classification loss for each example"""
        total_logprobs = []
        for i, example in enumerate(examples):
            example_logits = logits[i]
            # NOTE: we take the last element of the returned token list
            # this is because the tokenizer returns a 1-element list for GPT tokenizers
            # and a 2-element list with start token in the first position for OPT tokenizers
            class_tokens = [
                token[-1]
                for token in self.tokenizer(list(example.classes))["input_ids"]
            ]
            # log_softmax just subtracts a constant, so repeated applications change nothing
            # and there is no point in taking logprobs before focusing on the relevant indices
            example_logprobs = F.log_softmax(example_logits, dim=-1)
            relevant_logprobs = example_logprobs[class_tokens]
            total_logprobs.append(torch.logsumexp(relevant_logprobs, dim=-1).item())
        return total_logprobs

