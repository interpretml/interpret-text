## The below code is taken from https://github.com/naimenz/inverse-scaling-eval-pipeline

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from dataclasses import asdict, dataclass
import logging
import os
from typing_extensions import Literal
import requests
from datetime import timedelta
from typing import Optional, Union
from ratelimit import sleep_and_retry, limits


OPENAI_API_BASE_URL = "https://api.openai.com/v1/engines"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BaseGPT3List = [
    "ada",
    "babbage",
    "curie",
    "davinci",
]

BaseGPT3Model = Literal[
    "ada",
    "babbage",
    "curie",
    "davinci",
]

InstructGPT3List = [
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-001",
    "text-davinci-002",
    "text-davinci-003"
]

InstructGPT3Model = Literal[
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-001",
    "text-davinci-002",
    "text-davinci-003"
]

OpenaAIModelList = BaseGPT3List + InstructGPT3List
OpenAIModel = Literal[BaseGPT3Model, InstructGPT3Model]


@dataclass
class APIParameters:
    temperature: float = 0.0
    n: int = 1
    max_tokens: int = 1
    top_p: float = 1.0
    logprobs: Optional[int] = 100
    stop: Optional[list[str]] = None
    echo: bool = False

def call_api(
    prompt: Union[str, list[str]],
    model_name: OpenAIModel,
    api_params: Optional[APIParameters] = None,
) -> requests.Response:
    # dodgy error handling and retry code
    count = 0
    max_retries = 25
    while True:
        count += 1
        if count >= max_retries:
            raise ValueError(f"Retried too many times ({max_retries}), got error: {response_json['error']}")
        response = _call_api(prompt, model_name, api_params)
        response_json = response.json()
        if response.status_code != 200:
            logging.info(
                f"Retrying after error {response.status_code}: {response_json['error']}"
            )
        else:
            break
    return response


@sleep_and_retry
@limits(calls=50, period=timedelta(seconds=60).total_seconds())
def _call_api(
    prompt: Union[str, list[str]],
    model_name: OpenAIModel,
    api_params: Optional[APIParameters] = None,
) -> requests.Response:
    """This function makes the actual API call, and since we have a rate limit of 60 calls per minute,
    I will add rate limiting here (ideally we could increase the rate limit though)"""
    if api_params is None:
        api_params = APIParameters()
    # OpenAI gave my (Ian's) account the top 100 logprobs,
    # not just the top 5
    data = {
        "prompt": prompt,
        **asdict(api_params),
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    url = os.path.join(OPENAI_API_BASE_URL, model_name, "completions")
    response = requests.post(url, json=data, headers=headers)
    return response

template = """
Q: Convert "2 million" to a number.
A: 2000000

Q: Convert "a whole lot" to a number.
A: N/A

Q: Convert "{number_string}" to a number.
A:
""".strip()

parser_api_params = APIParameters(max_tokens=10, stop=["\n"])


class NumericParser(ABC):
    @abstractmethod
    def __call__(self, number_strings: list[str]) -> list[Optional[float]]:
        pass


class BasicParser(NumericParser):
    def __call__(self, number_strings: list[str]) -> list[Optional[float]]:
        prepped_strings = [prep_string(s) for s in number_strings]
        floats_from_parsed_strings: list[Optional[float]] = []
        for s in prepped_strings:
            try:
                parsed_s = float(s)
                floats_from_parsed_strings.append(parsed_s)
            except ValueError:
                floats_from_parsed_strings.append(None)

        return floats_from_parsed_strings


def prep_string(s: str) -> str:
    # clearing out leading/trailing spaces, commas, parentheses, and trailing full stops
    return s.strip().replace(",", "").replace(")", "").replace("(", "").rstrip(".")


class GPT3Parser(NumericParser):
    def __init__(self, model_name: InstructGPT3Model) -> None:
        self.model_name: InstructGPT3Model = model_name

    def __call__(self, number_strings: list[str]) -> list[Optional[float]]:
        filled_templates = [
            template.format(number_string=ns.strip()) for ns in number_strings
        ]
        response = call_api(filled_templates, self.model_name, parser_api_params)
        texts = [
            choice["text"].strip().replace(",", "")
            for choice in response.json()["choices"]
        ]
        print(f"strings returned from parser: {texts}")
        # check the response is non-empty and the model reports it found a valid number
        floats = [float(text) if text and text != "N/A" else None for text in texts]
        return floats
