import spacy
import html
import string
import numpy as np
import matplotlib.pyplot as plt
from spacy.lang.en import English
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display, HTML
from sklearn.feature_extraction.text import CountVectorizer


# Tokenizer is class instead of function to avoid multiple reloads of parser, stopwords and punctuation
# Uses spacy's inbuilt language tool for preprocessing in English [model](https://github.com/explosion/spaCy/tree/master/spacy/lang/en) 
class BOWTokenizer:
    def __init__(self, parser,
                 stop_words=spacy.lang.en.stop_words.STOP_WORDS,
                 punctuations=string.punctuation):
        self.parser = parser
        # list of stop words and punctuation marks
        self.stop_words = stop_words
        self.punctuations = punctuations

    def tokenize(self, sentence, keep_ids=False):
        EMPTYTOKEN = "empty_token"
        # Creating our token object, which is used to create documents with linguistic annotations.
        mytokens = self.parser(sentence)

        # Lemmatizing each token, removing blank space and converting each token into lowercase
        mytokens = [
            word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_
            for word in mytokens
        ]

        # Removing stop words
        if keep_ids is True:
            return [
                word
                if word not in self.stop_words and word not in self.punctuations
                else EMPTYTOKEN
                for word in mytokens
            ]
        else:
            return [
                word
                for word in mytokens
                if word not in self.stop_words and word not in self.punctuations
            ]

    def parse(self, sentence):
        return self.parser(sentence)


class BOWEncoder:
    def __init__(self):
        self.tokenizer = BOWTokenizer(
            English()
        )  # the tokenizer must have a tokenize() and parse() function.
        self.labelEncoder = LabelEncoder()
        self.vectorizer = CountVectorizer(
            tokenizer=self.tokenizer.tokenize, ngram_range=(1, 1)
        )
        self.decode_params = {}

    # The keep_ids flag, is used by explain local in the explainer to decode importances over raw features.
    def encode_features(self, X_str, needs_fit=True, keep_ids=False):
        # encoding while preserving ids, used only for importance computation
        # and not during training
        if keep_ids is True and isinstance(X_str, str):
            X_str = self.tokenizer.tokenize(X_str, keep_ids=True)
        # needs_fit will be set to true if encoder is not already trained
        if needs_fit is True:
            self.vectorizer.fit(X_str)
        if isinstance(X_str, str):
            X_str = [X_str]
        X_vec = self.vectorizer.transform(X_str)
        return [X_vec, self.vectorizer]

    def encode_labels(self, y_str, needs_fit=True):
        # TODO : add if statements for labels that are inputted as nd.arrays and lists.
        # convert from pandas dataframe to ndarray
        y_str = np.asarray(y_str[:]).reshape(-1, 1)
        if needs_fit is True:
            y_vec = self.labelEncoder.fit_transform(y_str)
        else:
            y_vec = self.labelEncoder.transform(y_str)
        return [y_vec, self.labelEncoder]

    def decode_imp(self, encoded_imp, input_text):
        EMPTYTOKEN = "empty_token"
        parsed_sentence = []
        # obtain parsed sentence, while preserving token -> position in sentence mapping
        for i in self.tokenizer.parse(input_text):
            parsed_sentence += [str(i)]
        encoded_text = self.tokenizer.tokenize(input_text, keep_ids=True)

        # replace words with an empty token if deleted when tokenizing
        encoded_word_ids = [
            None if word == EMPTYTOKEN else self.vectorizer.vocabulary_.get(word)
            for word in encoded_text
        ]
        # obtain word importance corresponding to the word vectors of the encoded sentence
        decoded_imp = [
            0 if idx is None else encoded_imp[idx] for idx in encoded_word_ids
        ]
        return (decoded_imp, parsed_sentence)


def plot_local_imp(parsed_sentence, word_importances, max_alpha=0.5):
    # Prevent special characters like & and < to cause the browser...
    # to display something other than what you intended.
    def html_escape(text):
        return html.escape(text)

    highlighted_text = []
    for i, word in enumerate(parsed_sentence):
        weight = word_importances[i]
        if weight > 0:
            highlighted_text.append(
                '<span style="background-color:rgba(135,206,250,'
                + str(abs(weight) / max_alpha)
                + ');">'
                + html_escape(word)
                + "</span>"
            )
        elif weight < 0:
            highlighted_text.append(
                '<span style="background-color:rgba(250,0,0,'
                + str(abs(weight) / max_alpha)
                + ');">'
                + html_escape(word)
                + "</span>"
            )
        else:
            highlighted_text.append(word)

    highlighted_text = " ".join(highlighted_text)
    display(HTML(highlighted_text))


def get_important_words(classifier, label_name, bow_encoder, clf_type="coef"):
    if clf_type == "coef":
        label_coefs_all = classifier.coef_
        # obtain label number / row corresponding to labelname
        label_row_num = bow_encoder.labelEncoder.transform([label_name])
        #convert from vector to scalar
        label_row_num = label_row_num[0]
        # obtain importance row corresponding to label number
        label_coefs = label_coefs_all[label_row_num, :]
    elif clf_type == "feature_importances":
        label_coefs = classifier.feature_importances_
    else:
        raise Exception("This feature is not yet supported.")
    # obtain feature ids of top labels sorted inascending order
    # use np.abs to obtain highest magnitude of importance, discarding directionality
    # np.argsort to ids corresponding to descending order of importances
    # np.flip to convert descending order to ascending order
    # TODO : Remove flatten below
    sorting_ids = (np.flip(np.argsort(np.abs(label_coefs))))
    top_ids = sorting_ids[0:20]  # view top 20 features per label
    # obtain raw words corresponding to top feature ids
    top_words = [bow_encoder.vectorizer.get_feature_names()[i] for i in top_ids]
    # obtain importance magnitudes corresponding to top feature ids
    top_importances = [label_coefs[i] for i in top_ids]
    return [top_words, top_importances]



def plot_global_imp(top_words, top_importances, label_name):
    plt.figure(figsize=(14, 7))
    plt.title("most important words for class label: " + str(label_name), fontsize=18)
    plt.bar(range(len(top_importances)), top_importances, color="r", align="center")
    plt.xticks(range(len(top_importances)), top_words, rotation=60, fontsize=18)
    plt.show()
