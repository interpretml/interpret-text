import logging
import time

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from spacy.lang.en import English
from utils_nlp.dataset.multinli import load_pandas_df

from interpret_text.utils.utils_bow import SpacyTokenizer,encode_labels
from interpret_text.utils.utils_bow import plot_local_imp, get_local_importances
from interpret_text.utils.utils_bow import get_important_words, plot_global_imp
from interpret_text.performance.utils.memory import get_peak_memory

from azureml.core.run import Run

logging.basicConfig(filename='benchmark.log')
test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.INFO)


azure_run = False

if azure_run:
    run = Run.get_context()
start = time.time()


#Run model
DATA_FOLDER = './temp'
TRAIN_SIZE = 0.3
TEST_SIZE = 0.2

df = load_pandas_df(DATA_FOLDER, "train")
df = df[df["gold_label"]=="neutral"]  # get unique sentences

# fetch documents and labels from data frame
X_str = df['sentence1'] # the document we want to analyze
ylabels = df['genre'] # the labels, or answers, we want to test against

# Convert tokens to BOW count vector
spacytokenizer = SpacyTokenizer(English())
countvectorizer = CountVectorizer(tokenizer = spacytokenizer.tokenize, ngram_range=(1,1))
X_vec = countvectorizer.fit_transform(X_str)

# Encode labels as numbers
labelencoder, ylabels = encode_labels(ylabels)

# Split into training and test set 
X_train, X_test, y_train, y_test = train_test_split(X_vec,ylabels, train_size = TRAIN_SIZE, 
                                                    test_size = TEST_SIZE, random_state=0)


tuned_parameters = [{'solver': ['saga'],'multi_class': ['multinomial'], 'C': [10**4]}]

model =  LogisticRegression()
classifier_CV = GridSearchCV(model, tuned_parameters, cv=3, scoring='accuracy')
classifier_CV.fit(X_train,y_train)


# obtain best classifier and hyper params
classifier = classifier_CV.best_estimator_
print("best classifier: " + str(classifier_CV.best_params_))

# obtain best classifier and hyper params
classifier = classifier_CV.best_estimator_
print("best classifier: " + str(classifier_CV.best_params_))

mean_accuracy = classifier.score(X_test, y_test, sample_weight=None)
print("accuracy = " + str(mean_accuracy*100) + "%")
y_pred = classifier.predict(X_test)
[precision, recall, fscore, support] = precision_recall_fscore_support(y_test, y_pred,average='macro')

print("The class names are as follows")
print(labelencoder.classes_)
label_name = "fiction"

#Obtain the top feature ids for the selected class label.           
#Map top features back to words.
top_words, top_importances = get_important_words(classifier, label_name, countvectorizer, labelencoder)
# #Plot the feature importances
# plot_global_imp(top_words, top_importances, label_name)


#Enter any document & label pair that needs to be interpreted
document = "I travelled to the beach. I took the train. I saw faries, dragons and elves"
label_name = "travel"

#Obtain the top feature ids for the selected class label
parsed_sentence, word_importances = get_local_importances(classifier, labelencoder, label_name,
                                                          document, spacytokenizer, countvectorizer)
#Visualize local feature importances as a heatmap over words in the document
# plot_local_imp(parsed_sentence, word_importances)


# Get elapsed execution time
end = time.time()
test_logger.info('elapsed time: ' + str(end - start))
peak_memory = get_peak_memory()
test_logger.info('peak memory usage: ' + str(peak_memory))
#log accuracy of scenario model (bert or logistic regression)

if azure_run:
    # log execution time and peak memory usage to the Azure Run
    run.log('execution time', end - start)
    run.log('peak memory usage', peak_memory)
