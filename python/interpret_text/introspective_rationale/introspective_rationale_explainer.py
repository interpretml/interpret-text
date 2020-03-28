from interpret_text.common.utils_classical import plot_local_imp
from interpret_text.introspective_rationale.introspective_rationale_components import (
    ClassifierModule,
    IntrospectionGeneratorModule,
    ClassifierWrapper,
)
from interpret_text.introspective_rationale.introspective_rationale_model import IntrospectiveRationaleModel
from interpret_text.explanation.explanation import _create_local_explanation
from interpret_text.common.utils_introspective_rationale import generate_data

import torch
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification

BERT_EMBEDDING_DIM = 768
RNN_EMBEDDING_DIM = 100


class IntrospectiveRationaleExplainer:
    """
    An explainer for training an explainable neural network for natural
    language processing and generating rationales used by that network.
    Based on the paper "Rethinking Cooperative Rationalization: Introspective
    Extraction and Complement Control" by Yu et. al.
    """

    def __init__(
        self,
        args,
        preprocessor,
        classifier_type="BERT",
        explainer=None,
        anti_explainer=None,
        generator=None,
        gen_classifier=None,
    ):
        """
        Initialize the IntrospectiveRationaleExplainer
        classifier type: {BERT, RNN, BERT-RNN, custom}
        If BERT, explainer, anti explainer, and generator classifier will
        be BERT modules
        If RNN, explainer, anti explainer, and generator classifier will
        be RNNs
        If BERT-RNN, generator classifier will be BERT module;
        explainer and anti explainer will be RNNs
        If custom, provide modules for explainer, anti_explainer, generator,
        and gen_classifier.
        """
        self.args = args

        if classifier_type == "BERT":
            args.gen_embedding_dim = BERT_EMBEDDING_DIM  # input dimension to use in the generator classifier
            args.bert_explainers = True
            args.embedding_dim = BERT_EMBEDDING_DIM  # input dimension to use in the estimator/anti-estimator classifiers
            args.hidden_dim = BERT_EMBEDDING_DIM  # input dimension to use in the hidden generator RNN
            self.explainer = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=args.num_labels,
                output_hidden_states=False,
                output_attentions=False,
            )
            self.anti_explainer = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=args.num_labels,
                output_hidden_states=False,
                output_attentions=False,
            )
            self.gen_classifier = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=args.num_labels,
                output_hidden_states=True,
                output_attentions=True,
            )
            self._freeze_classifier(self.explainer)
            self._freeze_classifier(self.anti_explainer)
            self._freeze_classifier(self.gen_classifier)
        elif classifier_type == "RNN":
            assert args.embedding_path is not None, \
                "embedding path must be specified if using RNN modules"
            args.bert_explainers = False
            args.gen_embedding_dim = RNN_EMBEDDING_DIM
            args.embedding_dim = RNN_EMBEDDING_DIM
            args.hidden_dim = RNN_EMBEDDING_DIM
            self.explainer = ClassifierModule(args, preprocessor.word_vocab)
            self.anti_explainer = ClassifierModule(args,
                                                   preprocessor.word_vocab)
            self.gen_classifier = ClassifierModule(args,
                                                   preprocessor.word_vocab)
        elif classifier_type == "BERT-RNN":
            assert args.embedding_path is not None, \
                "embedding path must be specified if using RNN modules"
            args.bert_explainers = False
            args.gen_embedding_dim = BERT_EMBEDDING_DIM
            args.embedding_dim = RNN_EMBEDDING_DIM
            args.hidden_dim = BERT_EMBEDDING_DIM
            self.explainer = ClassifierModule(args, preprocessor.word_vocab)
            self.anti_explainer = ClassifierModule(args,
                                                   preprocessor.word_vocab)
            self.gen_classifier = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=args.num_labels,
                output_hidden_states=True,
                output_attentions=True,
            )
            self._freeze_classifier(self.gen_classifier)
        else:
            assert args.hidden_dim is not None \
                and args.embedding_dim is not None,\
                "Hidden dim and embedding dim must be specified in args."
            assert explainer is not None\
                and anti_explainer is not None\
                and generator is not None\
                and gen_classifier is not None,\
                "Custom explainer, anti explainer, generator, and"\
                "generator classifier specifications are required."
            self.explainer = explainer
            self.anti_explainer = anti_explainer
            self.gen_classifier = gen_classifier

        self.generator = IntrospectionGeneratorModule(
            args, self.gen_classifier
        )

        self.model = IntrospectiveRationaleModel(
            args,
            preprocessor,
            self.explainer,
            self.anti_explainer,
            self.generator,
            self.gen_classifier,
        )
        self.labels = args.labels
        self.cuda = args.cuda
        if self.cuda:
            self.model.cuda()

    def _freeze_classifier(self, classifier, entire=False):
        """Freeze selected layers (or all of) a BERT classifier
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
        """Load a pretrained model in the explainer

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

    def train(self, *args, **kwargs):
        """Optionally pretrain the generator's introspective classifier, then
        train the explainer's model on the training data, with testing
        at the end of every epoch.
        """
        return self.fit(*args, **kwargs)

    def fit(self, df_train, df_test):
        """Optionally pretrain the generator's introspective classifier, then
        train the explainer's model on the training data, with testing
        at the end of every epoch.

        :param df_train: training data containing labels, lists of word token
            ids, pad/word masks, and token counts for each training example
        :type df_train: pd.DataFrame
        :param df_test: testing data containing labels, lists of word token
            ids, pad/word masks, and token counts for each testing example
        :type df_test: pd.DataFrame
        :return: the fitted model
        :rtype: IntrospectiveRationaleModel
        """

        if self.args.pretrain_cls:
            cls_wrapper = ClassifierWrapper(self.args, self.gen_classifier)
            cls_wrapper.fit(df_train, df_test)

            # freeze the generator's classifier entirely
            # (makes sense only if user wants to pretrain)
            if self.args.fixed_classifier:
                self._freeze_classifier(self.gen_classifier, entire=True)

        # train the three player model end-to-end
        self.model.fit(df_train, df_test)

        return self.model

    def predict(self, df_predict):
        """Generate rationales, predictions using rationales, predictions using
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
        batch_dict = generate_data(df_predict, self.args.cuda)
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
        """Calculate and store as model attributes:
        Average classification accuracy using rationales (self.avg_accuracy),
        Average classification accuracy rationale complements
            (self.anti_accuracy)
        Average sparsity of rationales (self.avg_sparsity)

        :param df_test: dataframe containing test data labels, tokens, masks,
            and counts
        :type df_test: pandas dataframe
        """
        self.model.test(df_test)

    def explain_local(
        self, sentence, preprocessor, hard_importances=False
    ):
        """Create a local explanation for a given sentence

        :param sentence: A segment of text
        :type sentence: string
        :param preprocessor: an intialized preprocessor to tokenize the
            given text with .preprocess() and .decode_single() methods
        :type preprocessor: Ex. GlovePreprocessor or BertPreprocessor
        :param hard_importances: whether to generate "hard" important/
            non-important rationales or float rationale scores, defaults
            to True
        :type hard_importances: bool, optional
        :return: local explanation object
        :rtype: DynamicLocalExplanation
        """
        # Pass in dummy ground truth label of 0 to run generate_data
        df_dummy_label = pd.DataFrame.from_dict({"labels": [0]})
        df_sentence = pd.concat(
            [df_dummy_label, preprocessor.preprocess([sentence.lower()])],
            axis=1
        )

        batch_dict = generate_data(df_sentence, self.args.cuda)
        x = batch_dict["x"]
        m = batch_dict["m"]
        predict_dict = self.predict(df_sentence)
        zs = predict_dict["rationale"]
        prediction = predict_dict["predict"]
        prediction_idx = prediction[0].max(0)[1]
        prediction = self.labels[prediction_idx]
        zs = np.array(zs.cpu())
        if not hard_importances:
            float_zs = self.model.get_z_scores(df_sentence)
            float_zs = float_zs[:, :, 1].detach()
            float_zs = np.array(float_zs.cpu())
            # set importances all words not selected as part of the rationale
            # to zero
            zs = zs * float_zs

        # generate human-readable tokens (individual words)
        seq_len = int(m.sum().item())
        ids = x[:seq_len][0]
        tokens = preprocessor.decode_single(ids)

        local_explanation = _create_local_explanation(
            classification=True,
            text_explanation=True,
            local_importance_values=zs.flatten(),
            method=str(type(self.model)),
            model_task="classification",
            features=tokens,
            classes=self.labels,
            predicted_label=prediction,
        )
        return local_explanation

    def visualize(self, local_explanation):
        """Create a heatmap of words important in a text given a
        local explanation

        :param local_explanation: local explanation object with
        "features" and "local importance values" attributes
        :type local_explanation: DynamicLocalExplanation
        """
        plot_local_imp(
            local_explanation._features,
            local_explanation._local_importance_values,
            max_alpha=0.5)
