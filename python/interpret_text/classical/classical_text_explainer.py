import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from interpret_text.common.utils_classical import plot_local_imp, plot_global_imp
from interpret_text.common.utils_classical import get_important_words, BOWEncoder
from interpret_text.common.constants import ExplainerParams
from interpret_text.explanation.explanation import _create_local_explanation
from interpret_community.explanation.explanation import _create_global_explanation

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
        """ Initialize the ClassicalTextExplainer
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
        """ Encode text strings in X_str as vectors.
        :param X_str: Strings to be encoded.
        :type X_str: array_like (array of strings, ndarray, pandas dataframe)
        :return: A model explanation object. It is guaranteed to be a LocalExplanation.
        :rtype: array_like (ndarray, pandas dataframe). Same rows as X_str
        """
        X_vec, _ = self.preprocessor.encode_features(X_str)
        return X_vec

    def train(self, *args, **kwargs):
        """  Wrapper function for 'fit()'. If the user wants to entirely swap out
            'fit()' with a customer trainer, they can modify train instead."""
        return self.fit(*args, **kwargs)

    def fit(self, X_str, y_train):
        """ Trains the model with training data and labels.
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

        # report metrics
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

    def explain_local(self, input_text, abs_sum_to_one=False):
        """ Returns an explanation object containing explanations over words
            in the input text string.
        :param input_text: String to be explained.
        :type input_text: str
        :return: A model explanation object containing importances and metadata.
        :rtype: LocalExplanation object
        """

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
        """ Returns list of top 20 features and corresponding importances over
            the whole dataset.
            * Obtain the top feature ids for the selected class label.
            * Map top features back to words.
            Can be rephrased in an intuitive sense as a local explanation over
            a sentence that contains every word in the vocabulary.
        :param label_name: Label for which importances are to be returned.
        :type label_name: str
        :return: A model explanation object containing importances and metadata.
        :rtype: global_explanation object
        *Note*: Label name is not applicable to tree based models. For them,
            the most important features are returned over all labels.
        *Note*: Edit get_important_words() function to change num of features
            to be something other than 20.
        """
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
        """ Wrapper function for plot_local_imp()
            Plots the top importances for a parsed sentence when corresponding
            importances are available.
            Internal fast prototyping tool for easy visualization.
            Serves as a visual proxy for dashboard."""
        plot_local_imp(parsed_sentence, word_importances, max_alpha=0.5)
