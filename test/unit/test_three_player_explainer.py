# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

# Unit tests for model explainability SDK. Doesn't test visualization method
import os
import torch
import pandas as pd

from interpret_text.three_player_introspective.three_player_introspective_explainer import (
    ThreePlayerIntrospectiveExplainer
)
from interpret_text.common.dataset.utils_sst2 import load_sst2_pandas_df
from interpret_text.common.utils_three_player import GlovePreprocessor, ModelArguments, load_glove_embeddings


class TestThreePlayerExplainer(object):
    def test_working(self):
        assert True

    def test_explain_three_player_model(self, tmp):
        # preparing inputs to explainer intialization
        CUDA = torch.cuda.is_available()
        LABEL_COL = "labels"
        TEXT_COL = "sentences"

        train_data = load_sst2_pandas_df('train')
        test_data = load_sst2_pandas_df('test')
        all_data = pd.concat([train_data, test_data])

        args = ModelArguments(CUDA, model_prefix="3PlayerModelRNN",
                              model_save_dir=tmp, save_best_model=False)
        args.embedding_path = load_glove_embeddings(tmp)
        args.labels = [0, 1]
        args.num_labels = 2

        token_count_thresh = 1
        max_sentence_token_count = 70
        preprocessor = GlovePreprocessor(all_data[TEXT_COL], token_count_thresh, max_sentence_token_count)

        # append labels to tokenizer output
        df_train = pd.concat([train_data[LABEL_COL], preprocessor.preprocess(train_data[TEXT_COL])], axis=1)
        df_test = pd.concat([test_data[LABEL_COL], preprocessor.preprocess(test_data[TEXT_COL])], axis=1)

        # initialize explainer
        explainer = ThreePlayerIntrospectiveExplainer(args, preprocessor, classifier_type="RNN")

        # testing fit
        explainer.fit(df_train, df_test)
        print(explainer.model.train_accs[-1])

        # testing train (should be same as fit)
        explainer = ThreePlayerIntrospectiveExplainer(args, preprocessor, classifier_type="RNN")
        explainer.train(df_train, df_test)
        print(explainer.model.train_accs[-1])

        # save model for later use in testing load_pretrained
        model_save_path = os.path.join(args.model_folder_path, args.model_prefix + "gen_classifier.pth")
        torch.save(
            explainer.model.state_dict(),
            model_save_path
        )

        # testing score
        explainer.score(df_test)
        print(explainer.model.avg_accuracy)
        print(explainer.model.test_accs[-1])
        print(explainer.model.avg_anti_accuracy)
        print(explainer.model.avg_sparsity)
        print(explainer.model.avg_continuity)

        # test load_pretrained by loading model
        explainer = ThreePlayerIntrospectiveExplainer(args, preprocessor, classifier_type="RNN")
        explainer.load_pretrained_model(model_save_path)
        # verify that it's the same model
        explainer.score(df_test)
        print(explainer.model.avg_accuracy)
        print(explainer.model.test_accs[-1])
        print(explainer.model.avg_anti_accuracy)
        print(explainer.model.avg_sparsity)
        print(explainer.model.avg_continuity)

        text = "Beautiful movie ; really good ; the popcorn was bad"
        # testing predict
        df_sentence = pd.concat(
            [pd.DataFrame.from_dict({"labels": [0]}), preprocessor.preprocess([sentence.lower()])],
            axis=1)
        predict_dict = self.predict(df_sentence)
        print(predict_dict)
        # rationale = predict_dict["rationale"]
        # prediction = predict_dict["predict"]
        # "anti_predict"
        # "cls_predict"

        # testing explain local
        local_explanation = explainer.explain_local(text, preprocessor, hard_importances=False)
        print(local_explanation.local_importance_values)
        # valid_imp_vals = np.array([
        #     0.2620866596698761,
        #     0.16004231572151184,
        #     0.17308972775936127,
        #     0.18205846846103668,
        #     0.26146841049194336,
        #     0.25957807898521423,
        #     0.3549807369709015,
        #     0.23873654007911682,
        #     0.2826242744922638,
        #     0.2700383961200714,
        #     0.3673151433467865,
        #     0.3899800479412079,
        #     0.20173774659633636,
        #     0.260466068983078,
        # ])
        # print(local_explanation.local_importance_values)
        # local_importance_values = np.array(local_explanation.local_importance_values)
        # cos_sim = dot(valid_imp_vals, local_importance_values) / (
        #     norm(valid_imp_vals) * norm(local_importance_values)
        # )
        # assert cos_sim >= 0.80



