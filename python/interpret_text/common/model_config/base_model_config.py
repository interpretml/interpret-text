from typing import Optional

from pydantic import BaseModel


class BaseModelConfig(BaseModel):
    """ Base Model config that will be used a model in a explainer"""
    cuda: bool
    pretrain_cls: Optional[bool]
    batch_size: Optional[int]
    num_epochs: Optional[int]
    num_pretrain_epochs: Optional[int]
    save_best_model: Optional[bool]
    model_save_dir: str
    model_prefix: str
    num_labels:int
