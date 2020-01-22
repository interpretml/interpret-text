from interpret_text.common.utils_classical import plot_local_imp
from interpret_text.common.constants import ExplainerParams
from interpret_text.three_player_introspective.three_player_introspective_model import ClassifierModule, IntrospectionGeneratorModule, ThreePlayerIntrospectiveModel
from interpret_text.explanation.explanation import _create_local_explanation

import os
import numpy as np
import pandas as pd

# from ThreePlayerIntrospectiveModel import ClassifierModule, IntrospectionGeneratorModule, ThreePlayerIntrospectiveModel

class ThreePlayerIntrospectiveExplainer:
    def __init__(self, args, word_vocab, explainer=ClassifierModule, anti_explainer=ClassifierModule, generator=IntrospectionGeneratorModule, classifier=ClassifierModule):
        '''
        explainer, anti_explainer, generator, and classifier: classes 
        that will be initialized based on parameters listed in args
        '''
        self.model = ThreePlayerIntrospectiveModel(args, word_vocab, explainer, anti_explainer, generator, classifier)
        self.labels = args.labels

    def train(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def fit(self, df_train, df_test, batch_size, num_iteration=40000, pretrain_cls=True):
        '''
        x_train: list of sentences (strings)
        y_train: list of labels (ex. 0 -- negative and 1 -- positive)
        '''
        # tokenize/otherwise process the lists of sentences
        if pretrain_cls:
            print('pre-training the classifier')
            self.model.pretrain_classifier(df_train, df_test, batch_size)
        # encode the list
        self.model.fit(df_train, batch_size, num_iteration)
        
        return self.model

    def predict(self, df_predict):
        self.model.eval()
        batch_x_, batch_m_, batch_y_ = self.model.generate_data(df_predict)
        predict, anti_predict, cls_predict, z, neg_log_prob = self.model.forward(batch_x_, batch_m_)
        predicts = predict.detach()
        anti_predict = anti_predict.detach()
        cls_predict = cls_predict.detach()
        z = z.detach()
        neg_log_prob = neg_log_prob.detach()
        return predict, anti_predict, cls_predict, z, neg_log_prob
    
    def score(self, df_test, test_batch_size=200):
        accuracy, anti_accuracy, sparsity = self.model.test(df_test, test_batch_size)
        return accuracy, anti_accuracy, sparsity

    def explain_local(self, sentence, label, tokenizer, hard_importances=True):
        df_label = pd.DataFrame.from_dict({"labels": [label]})
        df_sentence = pd.concat([df_label, tokenizer.tokenize([sentence.lower()])], axis=1)

        x, m, _ = self.model.generate_data(df_sentence)
        predict, _, _, zs, _ = self.predict(df_sentence)
        if not hard_importances:
            zs, _ = self.model.get_z_scores(df_sentence)
            predict_class_idx = np.argmax(predict.detach())
            zs = zs.detach()[:, :, predict_class_idx]

        zs = np.array(zs.tolist())

        # generate human-readable tokens (individual words)
        seq_len = int(m.sum().item())
        ids = x[:seq_len][0]
        tokens = [self.model.reverse_word_vocab[i.item()] for i in ids]

        local_explanation = _create_local_explanation(
            classification=True,
            text_explanation=False,
            local_importance_values=zs[0],
            method=str(type(self.model)),
            model_task="classification",
            features=tokens,
            classes=self.labels,
        )
    
        return local_explanation

    def visualize(self, word_importances, parsed_sentence):
        plot_local_imp(parsed_sentence, word_importances, max_alpha=.5)