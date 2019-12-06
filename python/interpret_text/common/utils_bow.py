import spacy
import html
import string
import numpy as np
import matplotlib.pyplot as plt
from spacy.lang.en import English
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display, HTML
from sklearn.feature_extraction.text import CountVectorizer
import warnings

# Tokenizer is class instead of function to avoid multiple reloads of parser, stopwords and punctuation
class BOWTokenizer:
    def __init__(self, parser):
        self.parser = parser
        # list of stop words and punctuation marks
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.punctuations = string.punctuation

    def tokenize(self, sentence, keep_ids = False):
        # Creating our token object, which is used to create documents with linguistic annotations.
        mytokens = self.parser(sentence)

        # Lemmatizing each token, removing blank space and converting each token into lowercase
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

        # Removing stop words
        if keep_ids is True:
            mytokens = [ word if word not in self.stop_words and word not in self.punctuations else "empty_token" for word in mytokens]
            return mytokens
        else:
            mytokens = [ word for word in mytokens if word not in self.stop_words and word not in self.punctuations ]
            return mytokens

    def parse(self, sentence):
        mytokens = self.parser(sentence)
        return mytokens

class BOWEncoder:
    def __init__(self):
        self.tokenizer = BOWTokenizer(English()) # the tokenizer must have a tokenize() and parse() function.
        self.labelEncoder = LabelEncoder()
        self.vectorizer = CountVectorizer(tokenizer = self.tokenizer.tokenize, ngram_range=(1,1))
        self.decode_params = {}

    # The keep_ids flag, is used by explain local in the explainer to decode importances over raw features.
    def encode_features(self, X_str, needs_fit = True, keep_ids = False):
        if keep_ids is True and isinstance(X_str,str):
            X_str = self.tokenizer.tokenize(X_str,keep_ids = True)
        if needs_fit is True:
            self.vectorizer.fit(X_str)
        #TODO : self.needs_fit = True <- why was this here ?
        if isinstance(X_str, str):
            X_str = [X_str]
        X_vec = self.vectorizer.transform(X_str)
        return [X_vec, self.vectorizer]


    def encode_labels(self, y_str,needs_fit = True):
        #TODO : add if statements for labels that are inputted as nd.arrays and lists.
        # convert from pandas dataframe to ndarray
        y_str = np.asarray(y_str[:]).reshape(-1,1)
        if needs_fit is True:
            y_vec = self.labelEncoder.fit_transform(y_str)
        else:
            y_vec = self.labelEncoder.transform(y_str)
        return [y_vec, self.labelEncoder]

    def decode_imp(self, encoded_imp, input_text):
        parsed_sentence = []
        for i in self.tokenizer.parse(input_text):
            parsed_sentence += [str(i)]
        encoded_text = self.tokenizer.tokenize(input_text, keep_ids = True)

        # replace words with an empty token if deleted when tokenizing
        encoded_word_ids = [None if word is 'empty_token' else self.vectorizer.vocabulary_.get(word) for word in encoded_text]
        # obtain word importance corresponding to the word vectors of the encoded sentence
        decoded_imp = [0 if idx == None else encoded_imp[0,idx] for idx in encoded_word_ids]
        return(decoded_imp, parsed_sentence)


def plot_local_imp(parsed_sentence, word_importances, max_alpha = 0.5):
    # Prevent special characters like & and < to cause the browser...
    # to display something other than what you intended.
    def html_escape(text):
        return html.escape(text)
    highlighted_text = []
    for i,word in enumerate(parsed_sentence):
        weight = word_importances[i]
        if weight > 0:
            highlighted_text.append('<span style="background-color:rgba(135,206,250,' + str(abs(weight) / max_alpha) +
                                    ');">' + html_escape(word) + '</span>')
        elif weight < 0:
            highlighted_text.append('<span style="background-color:rgba(250,0,0,' + str(abs(weight) / max_alpha) +
                                    ');">' + html_escape(word) + '</span>')
        else:
            highlighted_text.append(word)

    highlighted_text = ' '.join(highlighted_text)
    display(HTML(highlighted_text))

def get_important_words(classifier, label_name, bow_encoder, clf_type='coef'):
    if clf_type is'coef':
        label_coefs_all =  classifier.coef_
        label_coefs = label_coefs_all[bow_encoder.labelEncoder.transform([label_name]),:]
        sorting_ids = (np.flip(np.argsort(np.abs(label_coefs)))).flatten()
        top_ids = sorting_ids[0:20] # view top 20 features per label
        top_words = [bow_encoder.vectorizer.get_feature_names()[i] for i in top_ids]
        top_importances = [label_coefs[0,i] for i in top_ids]
        return [top_words, top_importances]
    else:
        raise Exception('This feature is not yet supported.')
        # TODO : Add support for sklearn classifiers that use feature importances instead

def plot_global_imp(top_words, top_importances, label_name):
    plt.figure(figsize = (14,7))
    plt.title("most important words for class label: " + str(label_name), fontsize = 18 )
    plt.bar(range(len(top_importances)), top_importances,
            color="r", align="center")
    plt.xticks(range(len(top_importances)), top_words, rotation=60, fontsize = 18)
    plt.show()