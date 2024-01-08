## The below code is taken from https://github.com/naimenz/inverse-scaling-eval-pipeline and modified.

from __future__ import annotations
from abc import ABC
import ast
from dataclasses import dataclass
from typing import Iterator
from typing_extensions import Literal
import pandas as pd

TaskType = Literal[
    "classification_acc",
    "classification_loss",
    "numeric",
    "sequence_prob",
    "logodds",
    "absolute_logodds",
]


@dataclass
class Example(ABC):
    prompt: str


@dataclass
class ExampleWithClasses(Example):
    prompt: str
    classes: tuple[str, ...]


@dataclass
class ClassificationExample(ExampleWithClasses):
    prompt: str
    classes: tuple[str, ...]
    answer_index: int


@dataclass
class NumericExample(Example):
    prompt: str
    true_answer: int
    anchor: int


@dataclass
class SequenceProbExample(Example):
    prompt: str
    completion: str


@dataclass
class LogoddsExample(ExampleWithClasses):
    prompt: str
    other_prompt: str
    classes: tuple[str, ...]
    answer_index: int


class Dataset:
    """Class to store examples to be run by HF or GPT3 models"""

    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def classification_from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = []
        for _, row in df.iterrows():
            # important to convert the string 'classes' back into a list
            classes_list = ast.literal_eval(str(row["classes"]))
            if any(cls[0] != ' ' for cls in classes_list):
                print(f"WARNING: some class label from {classes_list} does not have a leading space")
            example = ClassificationExample(
                prompt=row["prompt"],
                classes=classes_list,
                answer_index=row["answer_index"],
            )
            examples.append(example)
        return Dataset(examples)

    @classmethod
    def numeric_from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = []
        for _, row in df.iterrows():
            example = NumericExample(row["prompt"], row["true_answer"], row["anchor"])
            examples.append(example)
        return Dataset(examples)

    @classmethod
    def sequence_prob_from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = []
        for _, row in df.iterrows():
            example = SequenceProbExample(row["prompt"], row["completion"])
            examples.append(example)
        return Dataset(examples)

    @classmethod
    def logodds_from_df(cls, df: pd.DataFrame) -> Dataset:
        examples = []
        for _, row in df.iterrows():
            # important to convert the string 'classes' back into a list
            classes_list = ast.literal_eval(str(row["classes"]))
            if any(cls[0] != ' ' for cls in classes_list):
                print(f"WARNING: some class label from {classes_list} does not have a leading space")
            example = LogoddsExample(
                prompt=row["prompt"],
                other_prompt=row["other_prompt"],
                classes=classes_list,
                answer_index=row["answer_index"],
            )
            examples.append(example)
        return Dataset(examples)
