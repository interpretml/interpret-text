import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from interpret_text.common.utils_classical import plot_global_imp
from interpret_text.common.utils_classical import get_important_words, BOWEncoder
from interpret_text.common.constants import ExplainerParams
from interpret_text.explanation.explanation import _create_local_explanation
from interpret_community.explanation.explanation import _create_global_explanation

# BOW explainer class that allows extensibility to other encoders
# Uses logistic regression and 1-gram model by default


class ClassicalTextExplainer:
    def __init__(self, preprocessor=None, model=None,
                 is_trained=False, hyperparam_range=None):
        self.parsed_sentence = None
        self.word_importances = None
        self.model = model
        self.is_trained = is_trained
        if self.model is None and self.is_trained:
            raise Exception(
                "is_trained flag can't be set to true, if custom model not provided"
            )
        self.hyperparam_range = hyperparam_range
        if self.model is not None:
            # model is user defined
            if not self.is_trained and self.hyperparam_range is None:
                raise Exception(
                    "custom model needs to be supplied with custom hyperparameter range to search over"
                )
        self.preprocessor = BOWEncoder() if preprocessor is None else preprocessor

    def _encode(self, X_str):
        X_vec, _ = self.preprocessor.encode_features(X_str)
        return X_vec

    def train(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def fit(self, X_str, y_train):
        X_train = self._encode(X_str)
        if self.is_trained is False:
            if self.model is None:
                self.model = LogisticRegression()
                # Hyperparameters were chosen through hyperparamter optimization on MNLI
                self.hyperparam_range = [ExplainerParams.HYPERPARAM_RANGE]
            classifier_CV = GridSearchCV(
                self.model, self.hyperparam_range, cv=3, scoring="accuracy"
            )
            classifier_CV.fit(X_train, y_train)
            # set model as the best estimator from grid search results
            self.model = classifier_CV.best_estimator_
            best_params = classifier_CV.best_params_
        else:
            best_params = self.model.get_params()

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

    def explain_local(self, input_text, abs_sum_to_one=False):
        [encoded_text, _] = self.preprocessor.encode_features(
            input_text, needs_fit=False
        )
        encoded_label = self.model.predict(encoded_text)
        # convert from vector to scalar
        encoded_label = encoded_label[0]
        # Obtain the top feature ids for the selected class label
        if hasattr(self.model, "coef_"):
            # when #labels == 2, coef_ returns 1D array
            label_coefs_all = self.model.coef_
            if len(self.preprocessor.labelEncoder.classes_) == 2:
                label_coefs_all = np.vstack((-1 * label_coefs_all,
                                            label_coefs_all))
            encoded_imp = label_coefs_all[encoded_label, :]
        elif hasattr(self.model, "feature_importances_"):
            encoded_imp = self.model.feature_importances_
        else:
            raise Exception("model is missing coef_ or feature_importances_ attribute")
        decoded_imp, parsed_sentence_list = self.preprocessor.decode_imp(
            encoded_imp, input_text
        )

        if abs_sum_to_one is True:
            decoded_imp = decoded_imp / (np.sum(np.abs(decoded_imp)))

        local_explanantion = _create_local_explanation(
            classification=True,
            text_explanation=True,
            local_importance_values=np.array(decoded_imp),
            method=str(type(self.model)),
            model_task="classification",
            features=parsed_sentence_list,
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
