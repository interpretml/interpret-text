from interpret_text.common.utils_classical import plot_local_imp
from interpret_text.three_player_introspective.three_player_introspective_components import (
    ClassifierModule,
    IntrospectionGeneratorModule,
    ClassifierWrapper,
)
from interpret_text.three_player_introspective.three_player_introspective_model import ThreePlayerIntrospectiveModel
from interpret_text.explanation.explanation import _create_local_explanation
from interpret_text.common.utils_three_player import generate_data

import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification


class ThreePlayerIntrospectiveExplainer:
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
        Initialize the ThreePlayerIntrospectiveExplainer
        classifier type: {BERT, RNN, custom}
        If BERT, explainer, anti explainer, and generator classifier will
        be BERT modules
        If RNN, explainer, anti explainer, and generator classifier will
        be RNNs
        If custom, provide modules for explainer, anti_explainer, generator,
        and gen_classifier.
        """
        self.args = args
        if classifier_type == "BERT":
            self.BERT = True
            args.BERT = True
            args.embedding_dim = 768
            args.hidden_dim = 768
            self.explainer = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=2,
                output_hidden_states=False,
                output_attentions=False,
            )
            self.anti_explainer = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=2,
                output_hidden_states=False,
                output_attentions=False,
            )
            self.gen_classifier = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=2,
                output_hidden_states=True,
                output_attentions=True,
            )
        else:
            self.BERT = False
            args.BERT = False
            word_vocab = preprocessor.word_vocab
            if classifier_type == "RNN":
                self.explainer = ClassifierModule(args, word_vocab)
                self.anti_explainer = ClassifierModule(args, word_vocab)
                self.gen_classifier = ClassifierModule(args, word_vocab)
            else:
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

        self.model = ThreePlayerIntrospectiveModel(
            args,
            preprocessor,
            self.explainer,
            self.anti_explainer,
            self.generator,
            self.gen_classifier,
        )
        self.labels = args.labels

        if args.cuda:
            self.model.cuda()

    def freeze_bert_classifier(self, classifier, entire=False):
        if entire:
            for name, param in classifier.named_parameters():
                param.requires_grad = False
        else:
            for name, param in classifier.named_parameters():
                if "bert.embeddings" in name or ("bert.encoder" in name and
                                                 "layer.11" not in name):
                    param.requires_grad = False

    def train(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def fit(self, df_train, df_test):
        # tokenize/otherwise process the lists of sentences
        if self.BERT:
            self.freeze_bert_classifier(self.explainer)
            self.freeze_bert_classifier(self.anti_explainer)
            self.freeze_bert_classifier(self.gen_classifier)

        if self.args.pre_train_cls:
            cls_wrapper = ClassifierWrapper(self.args, self.gen_classifier)
            cls_wrapper.fit(df_train, df_test)

        if self.BERT:
            self.freeze_bert_classifier(self.gen_classifier, entire=True)

        # encode the list
        self.model.fit(df_train, df_test)

        return self.model

    def predict(self, df_predict):
        self.model.eval()
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
        self, sentence, label, preprocessor, hard_importances=True
    ):
        df_label = pd.DataFrame.from_dict({"labels": [label]})
        df_sentence = pd.concat(
            [df_label, preprocessor.preprocess([sentence.lower()])], axis=1
        )

        batch_dict = generate_data(df_sentence, self.args.cuda)
        x = batch_dict["x"]
        m = batch_dict["m"]
        predict_dict = self.predict(df_sentence)
        predict = predict_dict["predict"].cpu()
        zs = predict_dict["rationale"]
        if not hard_importances:
            zs = self.model.get_z_scores(df_sentence)
            predict_class_idx = np.argmax(predict)
            zs = zs[:, :, predict_class_idx].detach()

        zs = np.array(zs.cpu())

        # generate human-readable tokens (individual words)
        seq_len = int(m.sum().item())
        ids = x[:seq_len][0]
        tokens = [preprocessor.reverse_word_vocab[i.item()] for i in ids]

        local_explanation = _create_local_explanation(
            classification=True,
            text_explanation=True,
            local_importance_values=zs.flatten(),
            method=str(type(self.model)),
            model_task="classification",
            features=tokens,
            classes=self.labels,
        )

        return local_explanation

    def visualize(self, word_importances, parsed_sentence):
        plot_local_imp(parsed_sentence, word_importances, max_alpha=0.5)
