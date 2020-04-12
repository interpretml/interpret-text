from typing import Optional, Any

from interpret_text.experimental.common.model_config.base_model_config import BaseModelConfig


class IntrospectiveRationaleModelConfig(BaseModelConfig):
    """
    Model configuration used by Introspective Rationale Model
    """
    embedding_dim: Optional[int]
    hidden_dim: Optional[int]
    gen_embedding_dim: Optional[int]
    dropout_rate: Optional[float]
    layer_num: Optional[int]

    bert_explainers: Optional[bool]
    # used only by RNN
    embedding_path: Optional[str]

    label_embedding_dim: Optional[int]
    fixed_classifier: Optional[bool]

    # lambas for loss function
    lambda_sparsity: Optional[float]
    lambda_continuity: Optional[float]
    lambda_anti: Optional[float]

    # target_sparsity is the desired sparsity ratio
    target_sparsity: Optional[float]

    # this is the target number of target continuous pieces
    # it has no effect now, because lambda_continuity is 0.
    count_pieces: Optional[int]

    # rate at which the generator explores different rationales
    exploration_rate: Optional[float]

    # multiplier to reward/penalize an accuracy gap between the classifier
    # and anti-classifier
    lambda_acc_gap: Optional[float]

    # learning rate
    lr: Optional[float]

    # whether to tune the weights of the embedding layer in the RNN classifier module
    # many of these weights are from passed in glove embeddings, but
    # it could be useful to fine tune those weights that did not inherit glove vector values
    fine_tuning: Optional[bool]

    # training parameters
    train_batch_size: Optional[int]
    test_batch_size: Optional[int]

    # stop training if validation acc does not improve for more than
    # training_stop thresh
    training_stop_thresh: Optional[int]

    # the numerical labels for classification
    # ex: MNLI needs [0, 1, 2, 3, 4]
    labels: Optional[Any]

    # for saving models and logging
    save_best_model: Optional[bool]
    model_prefix: Optional[str]
    save_path: Optional[str]

    model_folder_path: Optional[str]
