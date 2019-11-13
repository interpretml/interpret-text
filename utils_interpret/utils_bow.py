# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file contains helper functions needed to run the bag of words baseline notebook

import spacy
import html
import string
import numpy as np
import matplotlib.pyplot as plt
from spacy.lang.en import English
from sklearn.preprocessing import LabelEncoder
from IPython.core.display import display, HTML

# spacy tokenizer class
# if keep_idx is true, allows inverse mapping back to text from tokens, for future text highlighting.
class SpacyTokenizer:
    def __init__(self, parser):
        self.parser = parser
        # list of stop words and punctuation marks
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.punctuations = string.punctuation

    def tokenize(self, sentence, keep_idx = False):
        # Creating our token object, which is used to create documents with linguistic annotations.
        mytokens = self.parser(sentence)

        # Lemmatizing each token, removing blank space and converting each token into lowercase
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

        # Removing stop words
        if keep_idx is True:
            mytokens = [ word if word not in self.stop_words and word not in self.punctuations else "empty_token" for word in mytokens]
            return mytokens
        else:
            mytokens = [ word for word in mytokens if word not in self.stop_words and word not in self.punctuations ]
            return mytokens

    def parse(self, sentence):
        mytokens = self.parser(sentence)
        return mytokens

def encode_labels(ylabels):
    labelencoder = LabelEncoder()
    # convert from pandas dataframe to ndarray
    ylabels = np.asarray(ylabels[:]).reshape(-1,1)
    ylabels = labelencoder.fit_transform(ylabels)
    return [labelencoder, ylabels]

def get_important_words(classifier, label_name, countvectorizer, labelencoder, clf_type='coef'):
    if clf_type is'coef':
        label_coefs_all =  classifier.coef_
        label_coefs = label_coefs_all[labelencoder.transform([label_name]),:]
        sorting_ids = (np.flip(np.argsort(np.abs(label_coefs)))).flatten()
        top_ids = sorting_ids[0:20] # view top 20 features per label
        top_words = [countvectorizer.get_feature_names()[i] for i in top_ids]
        top_importances = [label_coefs[0,i] for i in top_ids]
        return [top_words, top_importances]
    else:
        raise Exception('This feature is not yet supported.')
        # TODO : Add support for sklearn classifiers that use feature importances instead
        return

def plot_global_imp(top_words, top_importances, label_name):
    plt.figure(figsize = (14,7))
    plt.title("most important words for class label: " + str(label_name), fontsize = 18 )
    plt.bar(range(len(top_importances)), top_importances, 
            color="r", align="center")
    plt.xticks(range(len(top_importances)), top_words, rotation=60, fontsize = 18)
    plt.show()

def get_local_importances(classifier, labelencoder, label_name, document, spacytokenizer, countvectorizer):
    label_coefs = classifier.coef_[labelencoder.transform([label_name]),:]
    parsed_sentence = []
    for i in spacytokenizer.parse(document):
        parsed_sentence += [str(i)]
    doc_tokens = spacytokenizer.tokenize(document, keep_idx = True)
    word_ids = [None if word is 'empty_token' else countvectorizer.vocabulary_.get(word) for word in doc_tokens]
    word_importances = [0 if idx == None else label_coefs[0,idx] for idx in word_ids]
    return [parsed_sentence, word_importances]

def plot_local_imp(parsed_sentence, word_importances, max_alpha = 0.5):
    # Prevent special characters like & and < to cause the browser to display something other than what you intended.
    
    def html_escape(text):
        return html.escape(text)

    max_alpha = 0.5
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