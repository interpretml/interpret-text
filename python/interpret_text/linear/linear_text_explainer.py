import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from interpret_text.common.utils_bow import plot_local_imp, plot_global_imp
from interpret_text.common.utils_bow import get_important_words, BOWEncoder
from interpret_text.common.constants import ExplainerParams
from interpret_text.explanation.explanation import _create_local_explanation
from interpret_community.explanation.explanation import _create_global_explanation

# BOW explainer class that allows extensibility to other encoders
# Uses logistic regression and 1-gram model by default


class LinearTextExplainer:
    def __init__(self, preprocessor=None, model=None, hyperparam_range=None):
        self.parsed_sentence = None
        self.word_importances = None
        self.is_trained = False
        self.model = model
        self.hyperparam_range = hyperparam_range
        if self.model is not None and self.hyperparam_range is None:
            raise Exception(
                "custom model needs to be supplied with custom hyperparamter range to search over"
            )
        self.preprocessor = BOWEncoder() if preprocessor is None else preprocessor

    def _encode(self, X_str):
        X_vec, _ = self.preprocessor.encode_features(X_str)
        return X_vec

    def train(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def fit(self, X_str, y_train):
        X_train = self._encode(X_str)
        if self.model is None:
            self.model = LogisticRegression()
            # Hyperparameters were chosen through hyperparamter optimization on MNLI
            self.hyperparam_range = [ExplainerParams.HYPERPARAM_RANGE]
        elif self.model is not None and self.hyperparam_range is None:
            raise Exception(
                "custom model needs to be supplied with custom hyperparamter range to search over"
            )
        classifier_CV = GridSearchCV(
            self.model, self.hyperparam_range, cv=3, scoring="accuracy"
        )
        classifier_CV.fit(X_train, y_train)
        # set model as the best estimator from grid search results
        self.model = classifier_CV.best_estimator_
        best_params = classifier_CV.best_params_

        # report metrics
        class Encoder:
            def __init__(self, explainer):
                self._explainer = explainer

            def transform(self, X_test):
                X_vec, _ = self._explainer.preprocessor.encode_features(
                    X_test, needs_fit=False
                )
                return X_vec

            def fit(self, *args):
                return self

        text_model = Pipeline(
            steps=[("preprocessor", Encoder(self)), ("classifier", self.model)]
        )

        return [text_model, best_params]

    def explain_local(self, input_text):
        [encoded_text, _] = self.preprocessor.encode_features(
            input_text, needs_fit=False
        )
        encoded_label = self.model.predict(encoded_text)
        # convert from vector to scalar
        encoded_label = encoded_label[0]
        # Obtain the top feature ids for the selected class label
        if hasattr(self.model, "coef_"):
            encoded_imp = self.model.coef_[encoded_label, :]
        elif hasattr(self.model, "feature_importances_"):
            encoded_imp = self.model.feature_importances_
        else:
            raise Exception("model is missing coef_ or feature_importances_ attribute")
        decoded_imp, parsed_sentence = self.preprocessor.decode_imp(
            encoded_imp, input_text
        )
        local_explanantion = _create_local_explanation(
            classification=True,
            text_explanation=True,
            local_importance_values=np.array(decoded_imp),
            method=str(type(self.model)),
            model_task="classification",
            features=parsed_sentence,
            classes=self.preprocessor.labelEncoder.classes_,
        )
        return local_explanantion

    def explain_global(self, label_name):
        # Obtain the top feature ids for the selected class label.
        # Map top features back to words.
        clf_type = ""
        if hasattr(self.model, "coef_"):
            clf_type = "coef"
        elif hasattr(self.model, "feature_importances_"):
            clf_type = "feature_importances"
        else:
            raise Exception("model is missing coef_ or feature_importances_ attribute")
        top_words, top_importances = get_important_words(
            self.model, label_name, self.preprocessor, clf_type=clf_type
        )
        global_explanation = _create_global_explanation(
            global_importance_values=top_importances,
            features=top_words,
            method=str(type(self.model)),
            model_task="classification",
        )
        # Plot the feature importances
        plot_global_imp(top_words, top_importances, label_name)

        return global_explanation

    def visualize(self, word_importances, parsed_sentence):
        plot_local_imp(parsed_sentence, word_importances, max_alpha=0.5)
