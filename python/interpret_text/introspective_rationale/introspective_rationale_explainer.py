from typing import Iterable, Any, Optional, Union, Dict

import numpy as np
import pandas as pd
import torch
from interpret_text.common.base_explainer import BaseTextExplainer
from interpret_text.common.model_config.introspective_rationale_model_config import IntrospectiveRationaleModelConfig
from interpret_text.common.model_config.model_config_constants import get_bert_default_config, get_rnn_default_config, \
    get_bert_rnn_default_config
from interpret_text.common.preprocessor.bert_preprocessor import BertPreprocessor
from interpret_text.common.preprocessor.glove_preprocessor import GlovePreprocessor
from interpret_text.common.utils_classical import plot_local_imp
from interpret_text.common.utils_introspective_rationale import generate_data
from interpret_text.explanation.explanation import _create_local_explanation
from interpret_text.introspective_rationale.introspective_rationale_components import ClassifierWrapper, \
    ClassifierModule, IntrospectionGeneratorModule
from interpret_text.introspective_rationale.introspective_rationale_model import IntrospectiveRationaleModel
from transformers import BertForSequenceClassification

CLASSIFIER_TYPE_BERT = "BERT"
CLASSIFIER_TYPE_BERT_RNN = "BERT_RNN"
CLASSIFIER_TYPE_RNN = "RNN"
CLASSIFIER_TYPE_CUSTOM = "CUSTOM"


class IntrospectiveRationaleExplainer(BaseTextExplainer):
    """ Introspective Rationale Explainer
    """

    def __init__(self, classifier_type:
                 Union[CLASSIFIER_TYPE_RNN, CLASSIFIER_TYPE_BERT_RNN,
                       CLASSIFIER_TYPE_BERT, CLASSIFIER_TYPE_CUSTOM] = CLASSIFIER_TYPE_RNN, cuda=False):
        """ Initialize the explainer"""
        self.classifier_type = classifier_type
        self.cuda = cuda

    def load(self):
        """ Load explainer with default classifiers, generator based on prepackaged
        classifier option ['BERT', 'RNN', 'BERT_RNN']
        """
        self._load_classifier_modules()
        self._load_generator()
        self.freeze_classifier()
        self.set_model()

    def _load_classifier_modules(self):
        """ Load all classifier  modules based on prepackaged classifier option ['BERT', 'RNN', 'BERT_RNN']
        """
        self.set_classifier(self.classifier_type)
        self.set_anti_classifier(self.classifier_type)
        self.set_generator_classifier(self.classifier_type)

    def set_classifier(self, classifier_type=CLASSIFIER_TYPE_RNN, classifier=None):
        """ Set the classifier from prepackaged option or provide custom classifier

        :param classifier_type: One of ['BERT', 'RNN', 'BERT_RNN']
        :type: str
        :param classifier: Custom provided classifier
        :type: Any
        """
        if classifier_type == CLASSIFIER_TYPE_RNN or classifier_type == CLASSIFIER_TYPE_BERT_RNN:
            self.classifier = ClassifierModule(self.model_config, self.preprocessor.word_vocab)
        elif classifier_type == CLASSIFIER_TYPE_BERT:
            self.classifier = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=self.model_config.num_labels,
                output_hidden_states=False,
                output_attentions=False,
            )
        else:
            self.classifier = classifier

    def set_anti_classifier(self, classifier_type=CLASSIFIER_TYPE_RNN, anti_classifier=None):
        """ Set anti classifier from prepackaged option or provide custom anti classifier

        :param classifier_type: One of ['BERT', 'RNN', 'BERT_RNN']
        :type str
        :param anti_classifier: Custom provided anti classifier
        :type Any
        """
        if classifier_type == CLASSIFIER_TYPE_RNN or classifier_type == CLASSIFIER_TYPE_BERT_RNN:
            self.anti_classifier = ClassifierModule(self.model_config, self.preprocessor.word_vocab)

        elif classifier_type == CLASSIFIER_TYPE_BERT:
            self.anti_classifier = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=self.model_config.num_labels,
                output_hidden_states=False,
                output_attentions=False,
            )
        else:
            self.anti_classifier = anti_classifier

    def set_generator_classifier(self, classifier_type=CLASSIFIER_TYPE_RNN, generator_classifier=None):
        """ Set classifier for the Generator

        :param classifier_type: One of ['BERT', 'RNN', 'BERT_RNN']
        :type classifier_type: str
        :param generator_classifier: Custom provided classifier for generator
        :type generator_classifier: Any
        :return: Any
        """
        if classifier_type == CLASSIFIER_TYPE_RNN:
            self.generator_classifier = ClassifierModule(self.model_config, self.preprocessor.word_vocab)
        elif classifier_type == CLASSIFIER_TYPE_BERT or classifier_type == CLASSIFIER_TYPE_BERT_RNN:
            self.generator_classifier = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=self.model_config.num_labels,
                output_hidden_states=True,
                output_attentions=True,
            )
        else:
            self.generator_classifier = generator_classifier

    def get_generator_classifier(self):
        """ Get classifer for the generator in the model
        """
        return self.generator_classifier

    def set_generator(self):
        """ Set generator with generator classifier
        """
        self.generator = IntrospectionGeneratorModule(self.model_config, self.generator_classifier)

    def _load_generator(self):
        """ Load generator for this explainer
        """
        self.set_generator()

    def freeze_classifier(self):
        """ Freeze classifer modules for the explainer
        """
        if self.classifier_type == CLASSIFIER_TYPE_BERT:
            self._freeze_classifier(self.classifier)
            self._freeze_classifier(self.anti_classifier)
            self._freeze_classifier(self.generator_classifier)
        elif self.classifier_type == CLASSIFIER_TYPE_BERT_RNN:
            self._freeze_classifier(self.generator_classifier)
        else:
            # TODO: handle custom classifier modules
            pass

    def _set_model_config(self, model_config: Dict):
        """ Set model configuration used for training/fitting

        :param model_config: configurations used by prepackaged model
        :type model_config: Dict
        """
        self.model_config = IntrospectiveRationaleModelConfig.parse_obj(model_config)

    def get_model_config(self) -> IntrospectiveRationaleModelConfig:
        """ Return model config setup for this model

        :return: Model configuration
        :rtype: IntrospectiveRationaleModelConfig
        """
        return self.model_config

    def build_model_config(self, custom_config: Dict):
        """ Build model configuration provided through experiment and defaults for a classifier

        :param custom_config: configurations provided through experiment
        :type: Dict
        """
        config = {}
        config.update(self._load_defaults_model_config_dict())  # default merged into config
        config.update(custom_config)  # custom_config merged into config
        self._set_model_config(config)

    def _load_defaults_model_config_dict(self) -> Dict:
        """ Load default configuration for a given classifier

        :return: config: Model configuration
        :rtype: Dict
        """
        if self.classifier_type == CLASSIFIER_TYPE_BERT:
            config = get_bert_default_config()
        elif self.classifier_type == CLASSIFIER_TYPE_RNN:
            config = get_rnn_default_config()
        elif self.classifier_type == CLASSIFIER_TYPE_BERT_RNN:
            config = get_bert_rnn_default_config()
        else:
            # TODO: default configuration for a custom classifier
            config = {}
        return config

    def explain_local(self, text: str, **kwargs) -> _create_local_explanation:
        """ Create a local explanation for a given text
        :param text: A segment of text
        :type text: str
        :param kwargs:
                preprocessor: an intialized preprocessor to tokenize the
                given text with .preprocess() and .decode_single() methods
                preprocessor: Ex. GlovePreprocessor or BertPreprocessor
                hard_importances: whether to generate "hard" important/ non-important rationales
                or float rationale scores, defaults to True
                hard_importances: bool, optional
        :return: local explanation object
        :rtype: DynamicLocalExplanation
        """
        model_args = self.model_config
        df_dummy_label = pd.DataFrame.from_dict({"labels": [0]})
        df_sentence = pd.concat(
            [df_dummy_label, self.preprocessor.preprocess([text.lower()])],
            axis=1
        )
        batch_dict = generate_data(df_sentence, self.model_config.cuda)
        x = batch_dict["x"]
        m = batch_dict["m"]
        predict_dict = self.predict(df_sentence)
        zs = predict_dict["rationale"]
        prediction = predict_dict["predict"]
        prediction_idx = prediction[0].max(0)[1]
        prediction = model_args.labels[prediction_idx]
        zs = np.array(zs.cpu())
        if not kwargs['hard_importances']:
            float_zs = self.model.get_z_scores(df_sentence)
            float_zs = float_zs[:, :, 1].detach()
            float_zs = np.array(float_zs.cpu())
            # set importances all words not selected as part of the rationale
            # to zero
            zs = zs * float_zs
            # generate human-readable tokens (individual words)
            seq_len = int(m.sum().item())
            ids = x[:seq_len][0]
        tokens = kwargs['preprocessor'].decode_single(ids)

        local_explanation = _create_local_explanation(
            classification=True,
            text_explanation=True,
            local_importance_values=zs.flatten(),
            method=str(type(self.model)),
            model_task="classification",
            features=tokens,
            classes=model_args.labels,
            predicted_label=prediction,
        )
        return local_explanation

    def fit(self, X: Iterable[Any], y: Optional[Iterable[Any]]) -> IntrospectiveRationaleModel:
        """ Optionally pretrain the generator's introspective classifier, then
        train the explainer's model on the training data, with testing
        at the end of every epoch.

        :param X: training data containing labels, lists of word token
        ids, pad/word masks, and token counts for each training example
        :type X: pd.DataFrame
        :param y: testing data containing labels, lists of word token
            ids, pad/word masks, and token counts for each testing example
        :type y: pd.DataFrame
        :return: fitted model
        :rtype IntrospectiveRationaleModel
        """
        if self.model_config.pretrain_cls:
            cls_wrapper = ClassifierWrapper(self.model_config, self.generator_classifier)
            cls_wrapper.fit(X, y)

            # freeze the generator's classifier entirely
            # (makes sense only if user wants to pretrain)
            if self.model_config.fixed_classifier:
                self._freeze_classifier(self.generator_classifier, entire=True)

        # train the three player model end-to-end
        self.model.fit(X, y)

        return self.model

    def train(self, *args, **kwargs):
        """ Optionally pretrain the generator's introspective classifier, then
        train the explainer's model on the training data, with testing
        at the end of every epoch.
        """
        return self.fit(*args, **kwargs)

    def set_model(self):
        """ Set model for the explainer
        """
        self.model = IntrospectiveRationaleModel(
            self.model_config,
            self.preprocessor,
            self.classifier,
            self.anti_classifier,
            self.generator,
            self.generator_classifier,
        )
        if self.cuda:
            self.model.cuda()

    def get_model(self) -> IntrospectiveRationaleModel:
        """ Get model for this explainer
        """
        return self.model

    def set_preprocessor(self, preprocessor: Union[GlovePreprocessor, BertPreprocessor]):
        """ Set processor for this explainer

        :param preprocessor: preprocessor
        :type Union[GlovePreprocessor, BertPreprocessor]
        """
        self.preprocessor = preprocessor

    def get_preprocessor(self) -> Union[GlovePreprocessor, BertPreprocessor]:
        """ Get preprocessor of this explainer """
        return self.preprocessor

    def _freeze_classifier(self, classifier, entire=False):
        """ Freeze selected layers (or all of) a BERT classifier
        """
        if entire:
            for name, param in classifier.named_parameters():
                param.requires_grad = False
        else:
            for name, param in classifier.named_parameters():
                if "bert.embeddings" in name or ("bert.encoder" in name
                                                 and "layer.11" not in name):
                    param.requires_grad = False

    def load_pretrained_model(self, pretrained_model_path):
        """ Load a pretrained model in the explainer

        :param pretrained_model_path: a path to a saved torch state dictionary
        :type pretrained_model_path: string
        :return: the pretrained model
        :rtype: IntrospectiveRationaleModel
        """
        if self.cuda:
            self.model.load_state_dict(torch.load(pretrained_model_path))
        else:
            self.model.load_state_dict(torch.load(pretrained_model_path,
                                                  map_location='cpu'))
        self.model.eval()
        return self.model

    def predict(self, df_predict):
        """ Generate rationales, predictions using rationales, predictions using
        anti-rationales (complement of generated rationales), and introspective
        generator classifier predictions for given examples.

        :param df_predict: data containing labels, lists of word token
            ids, pad/word masks, and token counts for each testing example
        :type df_predict: pd.DataFrame
        :return: Dictionary with fields:
            "predict": predictions using generated rationales
            "anti_predict": predictions using complements of generated
                rationales
            "cls_predict": predictions from introspective generator,
            "rationale": mask indicating whether words were used in rationales,
        :rtype: dict
        """
        self.model.eval()
        self.model.training = False
        batch_dict = generate_data(df_predict, self.model_config.cuda)
        batch_x_ = batch_dict["x"]
        batch_m_ = batch_dict["m"]
        forward_dict = self.model.forward(batch_x_, batch_m_)
        predict = forward_dict["predict"]
        anti_predict = forward_dict["anti_predict"]
        cls_predict = forward_dict["cls_predict"]
        z = forward_dict["z"]
        predict = predict.detach()
        anti_predict = anti_predict.detach()
        cls_predict = cls_predict.detach()
        z = z.detach()
        predict_dict = {
            "predict": predict,
            "anti_predict": anti_predict,
            "cls_predict": cls_predict,
            "rationale": z,
        }
        self.model.training = True
        return predict_dict

    def score(self, df_test):
        """ Calculate and store as model attributes:
        Average classification accuracy using rationales (self.avg_accuracy),
        Average classification accuracy rationale complements
            (self.anti_accuracy)
        Average sparsity of rationales (self.avg_sparsity)
        :param df_test: dataframe containing test data labels, tokens, masks,
            and counts
        :type df_test: pandas dataframe
        """
        self.model.test(df_test)

    def visualize(self, local_explanation):
        """ Create a heatmap of words important in a text given a local explanation

        :param local_explanation: local explanation object with
        "features" and "local importance values" attributes
        :type local_explanation: DynamicLocalExplanation
        """
        plot_local_imp(
            local_explanation._features,
            local_explanation._local_importance_values,
            max_alpha=0.5)
