from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from interpret_text.common.utils_bow import plot_local_imp, plot_global_imp
from interpret_text.common.utils_bow import get_important_words, BOWEncoder


# BOW explainer class that allows extensibility to other encoders
# Uses logistic regression and 1-gram model by default
class BOWExplainer:
    def __init__(self, model=None, hyperparam_range=None):
        self.parsed_sentence = None
        self.word_importances = None
        self.is_trained = False
        self.model = model
        if self.model is not None and self.hyperparam_range is None:
            raise Exception(
                "custom model needs to be supplied with custom hyper paramter range to search over"
            )
        self.preprocessor = BOWEncoder()

    def encode(
        self,
        X_str,
        y_str,
        train_test_split_params={"train_size": 0.8, "test_size": 0.2},
    ):
        [X_vec, _] = self.preprocessor.encode_features(X_str)
        [y_vec, _] = self.preprocessor.encode_labels(y_str)
        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y_vec, **train_test_split_params
        )
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        if self.model is None:
            self.model = LogisticRegression()
            # Hyper parameters were chosen through hyperparamter optimization on MNLI
            hyperparam_range = [
                {"solver": ["saga"], "multi_class": ["multinomial"], "C": [10 ** 4]}
            ]
        classifier_CV = GridSearchCV(
            self.model, hyperparam_range, cv=3, scoring="accuracy"
        )
        classifier_CV.fit(X_train, y_train)
        # set model as the best estimator from grid search results
        self.model = classifier_CV.best_estimator_
        best_params = classifier_CV.best_params_
        # report metrics
        mean_accuracy = classifier_CV.best_score_
        print("mean_accuracy = " + str(mean_accuracy * 100) + "%")
        return [self.model, best_params]

    def explain_local(self, input_text):
        [encoded_text, _] = self.preprocessor.encode_features(
            input_text, needs_fit=False
        )
        label_name = self.model.predict(encoded_text)
        # TODO : Vectorize list comprehensions to speed up word importance finding process
        # Obtain the top feature ids for the selected class label
        [encoded_label, _] = self.preprocessor.encode_labels([label_name])
        encoded_imp = self.model.coef_[encoded_label, :]
        decoded_imp, parsed_sentence = self.preprocessor.decode_imp(
            encoded_imp, input_text
        )
        return (decoded_imp, parsed_sentence)

    def explain_global(self, label_name):
        # Obtain the top feature ids for the selected class label.
        # Map top features back to words.
        top_words, top_importances = get_important_words(
            self.model, label_name, self.preprocessor, clf_type="coef"
        )
        # Plot the feature importances
        plot_global_imp(top_words, top_importances, label_name)
        return [top_words, top_importances]

    def visualize(self, word_importances, parsed_sentence):
        plot_local_imp(parsed_sentence, word_importances, max_alpha=0.5)
