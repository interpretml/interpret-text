# TODO: Refactor to use BaseTextExplainer (see IntrospectiveRationaleExplainer)

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from interpret_text.experimental.common.base_explainer import _validate_X
from interpret_text.experimental.common.utils_classical import BOWEncoder
from interpret_text.experimental.common.constants import ExplainerParams
from interpret_text.experimental.explanation import _create_local_explanation

# BOW explainer class that allows extensibility to other encoders
# Uses logistic regression and 1-gram model by default


class ClassicalTextExplainer:
    """The ClassicalTextExplainer for returning explanations for n-gram
    bag-of-words models using sklearn's classifier API.
    Also serves as extensible wrapper with components built to support addition
    of fresh explainers using certain parts of this class.
    """

    def __init__(self, preprocessor=None, model=None,
                 hyperparam_range=None, is_trained=False):
        """Initialize the ClassicalTextExplainer
        :param model: Linear models with linear coefs mapped to features or
            tree based models with inbuilt feature_importances that are sklearn
            models or follow the sklearn API.
        :type: sklearn.ensemble or sklearn.linear_model (natively supported)
        :param preprocessor: Custom preprocessor for encoding text into vector
            form. Contains custom parser, vectorizer and tokenizer. Reference
            utils_classical.BOWEncoder for preprocessor's API template.
        :type preprocessor: object
        :param hyperparam_range: Custom hyper parameter range to search over
            when training input model. passed to sklearn's GridsearchCV's
            as param_grid argument.
        :type hyperparam_range: dict     """
        self.parsed_sentence = None
        self.word_importances = None
        self.model = model
        self.is_trained = is_trained
        if self.model is None and self.is_trained:
            raise Exception(
                "Is_trained flag can't be set to true, if custom model not provided."
            )
        self.hyperparam_range = hyperparam_range
        if self.model is not None:
            # model is user defined
            if not self.is_trained and self.hyperparam_range is None:
                raise Exception(
                    "Custom model needs to be supplied with custom hyperparameter range to search over."
                )
        self.preprocessor = BOWEncoder() if preprocessor is None else preprocessor

    def _encode(self, X_str):
        """Encode text strings in X_str as vectors.
        :param X_str: Strings to be encoded.
        :type X_str: array_like (array of strings, ndarray, pandas dataframe)
        :return: A model explanation object. It is guaranteed to be a LocalExplanation.
        :rtype: array_like (ndarray, pandas dataframe). Same rows as X_str
        """
        X_vec, _ = self.preprocessor.encode_features(X_str)
        return X_vec

    def train(self, *args, **kwargs):
        """Wrapper function for 'fit()'. If the user wants to entirely swap out
            'fit()' with a customer trainer, they can modify train instead.
        :return: A model explanation object. It is guaranteed to be a LocalExplanation.
        :return: List of length 2 . The elements are:
            * An sklearn pipeline object containing trained encoder and trained model.
            * Dict containing mapping from features to the best hyperparameters.
        :rtype: list"""
        return self.fit(*args, **kwargs)

    def fit(self, X_str, y_train):
        """Trains the model with training data and labels.
            Includes:
            * Encoding X_str into vector form.
            *Note*: y_train is assumed to be encoded into sklearn compatible format.
            (use sklearn's label encoder for this purpose externally)
            * Training the model.
            * Grid search over parameter range.
            * Returns best model and corresponding hyper parameters.
        :param X_str: Dataset of strings to train on.
        :type X_str: array_like (array of strings, ndarray, pandas dataframe)
        :param y_train: Labels in encoded vector form directly sent to model.fit().
        :type y_train: array_like
        :return: List of length 2 . The elements are:
            * An sklearn pipeline object containing trained encoder and trained model.
            * Dict containing mapping from features to the best hyperparameters.
        :rtype: list
        """
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

        class Encoder:
            """Encoder object recast into an API that is compatible with
            Pipeline().
            """
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

    def explain_local(self, X, y=None, name=None):
        """Returns an explanation object containing explanations over words
            in the input text string.
        :param X: String to be explained.
        :type X: str
        :param y: The ground truth label for the sentence
        :type y: string
        :param name: a name for saving the explanation, currently ignored
        :type str
        :return: A model explanation object containing importances and metadata.
        :rtype: LocalExplanation
        """
        X = _validate_X(X)

        [encoded_text, _] = self.preprocessor.encode_features(
            X, needs_fit=False
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
            encoded_imp, X
        )

        local_explanantion = _create_local_explanation(
            classification=True,
            text_explanation=True,
            local_importance_values=np.array(decoded_imp),
            method=str(type(self.model)),
            model_task="classification",
            features=parsed_sentence_list,
            classes=self.preprocessor.labelEncoder.classes_,
            true_label=y
        )
        return local_explanantion
