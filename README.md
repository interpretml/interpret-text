
# Interpret-Text - Alpha Release package
<img alt="PyPI" src="https://img.shields.io/pypi/v/interpret-text"> <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/interpret-text"> <img alt="GitHub" src="https://img.shields.io/github/license/interpretml/interpret-text"> [![Build Status](https://dev.azure.com/responsibleai/interpret-text/_apis/build/status/CPU%20unit%20tests%20-%20linux?branchName=master&jobName=cpu_unit_tests_linux)](https://dev.azure.com/responsibleai/interpret-text/_build/latest?definitionId=62&branchName=master) 
    [![Build Status](https://dev.azure.com/responsibleai/interpret-text/_apis/build/status/CPU%20integration%20tests%20-%20linux?branchName=master&jobName=cpu_integration_tests_linux)](https://dev.azure.com/responsibleai/interpret-text/_build/latest?definitionId=61&branchName=master)



Interpret-Text builds on [Interpret](https://github.com/interpretml/interpret), an open source python package for training interpretable models and helping to explain blackbox machine learning systems. We have added extensions to support text models.

This repository contains an SDK and example Jupyter notebooks to showcase its use.

# Contents

-  [Overview of Interpret-Text](#overview)
-  [Target Audience](#target-audience)
-  [Getting Started](#getting-started)
-  [Supported NLP Scenarios](#models)
-  [Supported Explainers](#explainers)
-  [Use Interpret-Text](#use)
-  [Contributing](#contrib)
-  [Code of Conduct](#code)

<a  name="overview"></a>

# Overview of Interpret-Text
Interpret-Text incorporates community developed interpretability techniques for NLP models and a visualization dashboard to view the results. Users can run their experiments across multiple state-of-the-art explainers and easily perform comparative analysis on them. Using these tools, users will be able to explain their machine learning models globally on each label or locally for each document. In particular, this open-source toolkit:
1. Actively incorporates innovative text interpretability techniques, and allows the community to further expand its offerings
2. Creates a common API across the integrated libraries
3. Provides an interactive visualization dashboard to empower its users to gain insights into their data

<a  name="target-audience"></a>

# Target Audience

1. Developers/Data Scientists: Having all of the interpretability techniques in one place makes it easy for data scientists to experiment with different interpretability techniques and explain their model in a scalable and seamless manner. The set of rich interactive visualizations allow developers and data scientists to train and deploy more transparent machine learning models instead of wasting time and effort on generating customized visualizations, addressing scalability issues by optimizing third-party interpretability techniques, and adopting/operationalizing interpretability techniques.

2. Business Executives: The core logic and visualizations are beneficial for raising awareness among those involved in developing AI applications, allow them to audit model predictions for potential unfairness, and establish a strong governance framework around the use of AI applications.

3. Machine Learning Interpretability Researchers: Interpret's extension hooks make it easy to extend, meaning interpretability researchers who are interested in adding their own techniques can easily add them to the community repository and compare it to state-of-the-art and proven interpretability techniques and/or other community techniques.


<a  name="getting-started"></a>

# Getting Started

This repository uses Anaconda to simplify package and environment management.

To setup on your local machine:

<details>

<summary><strong><em>1. Clone the interpret-text repository</em></strong></summary>

Clone and cd into the repository
```
git clone https://github.com/interpretml/interpret-text.git
cd interpret-text
```
</details>

<details><summary><strong><em>2. Set up Environment</em></strong></summary>

    a. Install Anaconda with Python >= 3.7 
       [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) is a quick way to get started.


<details><summary><strong><em>2.1 Create and activate conda environment (For CPU): </strong></em></summary>

```
    python tools/generate_conda_files.py
    conda env create -n interpret_cpu --file=interpret_cpu.yaml
    conda activate interpret_cpu
```
</details>

<details><summary><strong><em>2.2 Create and activate conda environment (For GPU): </em></strong></summary>

```
    python tools/generate_conda_files.py --gpu
    conda env create -n interpret_gpu --file=interpret_gpu.yaml
    conda activate interpret_gpu
```
</details>
</details>

<details>
<summary><strong><em>3. Install package </em></strong></summary>

You can install the package from source or from pipy.

<details><summary><strong><em>3.1 From source (developers): </strong></em></summary>

Run the below commands from ```interpret-text/python```

```
    pip install -e .
    jupyter nbextension install interpret_text.widget --py --sys-prefix 
    jupyter nbextension enable interpret_text.widget --py --sys-prefix
```
</details>

<details><summary><strong><em>3.2 From github (package users): </strong></em></summary>

```
    pip install interpret-text
```
</details>

</details>

<details>
<summary><strong><em>4. Set up and run Jupyter Notebook server </em></strong></summary>

Install and run Jupyter Notebook
```
    pip install notebook
    jupyter notebook
```
</details>

# <a name="models"></a>

# Supported NLP Scenarios

Currently this repository only provides support for the text classification scenario.
# <a name="explainers"></a>

# Supported Explainers
The following is a list of the explainers available in this repository:
* Classical Text Explainer - (Default: [Bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) with Logistic Regression)

* [Unified Information Explainer](https://www.microsoft.com/en-us/research/publication/towards-a-deep-and-unified-understanding-of-deep-neural-models-in-nlp/)

* [Introspective Rationale Explainer](http://people.csail.mit.edu/tommi/papers/YCZJ_EMNLP2019.pdf)

## Explanation Method Comparison Chart
|  | Classical Text Explainer | Unified Information Explainer | Introspective Rationale Explainer |
|---------------|---------|:-------------------:|:----------------------------:|
| Input model support | Scikit-learn linear models and tree-based models | PyTorch | PyTorch |
| Explain BERT | No | Yes  | Yes  |
| Explain RNN  | No | No | Yes |
| NLP pipeline support | Handles text pre-processing, encoding, training, hyperparameter tuning | Uses BERT tokenizer however user needs to supply trained/fine-tuned BERT model, and samples of trained data | Generator and predictor modules handle the required text pre-processing.
| Sample notebook | [Classical Text Explainer Sample Notebook](https://nbviewer.jupyter.org/github/interpretml/interpret-text/blob/master/notebooks/text_classification/text_classification_classical_text_explainer.ipynb) | [Unified Information Explainer Sample Notebook](https://nbviewer.jupyter.org/github/interpretml/interpret-text/blob/master/notebooks/text_classification/text_classification_unified_information_explainer.ipynb) | [Introspective Rationale Explainer Sample Notebook](https://nbviewer.jupyter.org/github/interpretml/interpret-text/blob/master/notebooks/text_classification/text_classification_introspective_rationale_explainer.ipynb)|

## Classical Text Explainer

The ClassicalTextExplainer extends text interpretability to classical machine learning models. 
This [notebook](notebooks/text_classification/text_classification_classical_text_explainer.ipynb) provides a step by step walkthrough of operationalizing the ClassicalTextExplainer in an ML pipeline.

### Preprocessing and the Pipeline:

The ClassicalTextExplainer serves as a high level wrapper for the entire NLP pipeline, by natively handling the text preprocessing, encoding, training and hyperparameter optimization process. 
This allows the user to supply the dataset in text form without need for any external processing, with the entire text pipeline process being handled by the explainer under the hood.                         

In its default configuration the preprocessing pipeline uses a 1-gram bag-of-words encoder implemented by sklearn's count-vectorizer. The [utilities](python/interpret_text/experimental/common/utils_classical.py) file contains the finer details of the preprocessing steps in the default pipeline.            

### Supported Models:

The ClassicalTextExplainer natively supports 2 families of models: 

* Linear models with support for a '*coefs_*' call under sklearn's linear_model module 
* Tree based models with a 'feature_importances' call under sklearn's ensemble module  

In the absence of a user supplied model, the ClassicalTextExplainer defaults to sklearn's logistic regression.
In addition to the above mentioned models, any model that follows the same API layout and is compatible with sparse representations as input will also be supported.
Apart from Logistic regression, we have successfully tested the framework with [LightGBM](https://github.com/microsoft/LightGBM) and Random Forests as well.

### Extensibility and Modularity:

The ClassicalTextExplainer has been designed with explicit intent of being modular and extensible.

The API allows for users to swap out nearly every component including the preprocessor, tokenizer, model and training routine with varying levels of difficulty. The API is composed such that a modified explainer would still be able to leverage the rest of the tooling implemented within the package.

The text encoding and decoding components are both closely tied to each other. Should the user wish to use a custom encoding process, it has to come paired with its own custom decoding process.

### Explainability:

The ClassicalTextExplainer offers a painfree API to surface explanations inherent to supported models. The natively supported linear models such as linear regression and logisitic regression are considered to be glass-box explainers. A glass-box explainer implies a model that is innately explainable, where the user can fully observe and dissect the process adopted by the model in making a prediction. The family of linear models such as logistic regression and ensemble methods like random forests can be considered to be under the umbrella of glass-box explainers. Neural networks and Kernel based models are usually not considered glass-box models.

By default, the ClassicalTextExplainer leverages this inherent explainability by exposing weights and importances over encoded tokens as explanations over each word in a document. In practice, these can be accessed through the visualization dashboard or the explanation object.

The explanations provided by the aforementioned glass-box methods serve as direct proxies for weights and parameters in the model, which make the final prediction. This allows us to have high confidence in the correctness of the explanation and strong belief in humans being able to understand the internal configuration of the trained machine learning model.

If the user supplies a custom model, the nature of their model explanability (glass-box , grey-box, black-box) will carry over to importances produced by the explainer as well.


## Unified Information Explainer
The UnifiedInformationExplainer uses an information-based measure to provide unified and coherent explanations on the intermediate layers of deep NLP models. While this model can explain various deep NLP models, we only implement text interpretability for BERT here. This [notebook](notebooks/text_classification/text_classification_unified_information_explainer.ipynb) provides an example of how to load and preprocess data and retrieve explanations for all the layers of BERT - the transformer layers, pooler, and classification layer.

### Preprocessing:
The UnifiedInformationExplainer handles the required text pre-processing. Each sentence is tokenized using the `BERT Tokenizer`.

### Supported Models:
The UnifiedInformationExplainer only supports BERT at this time. A user will need to supply a trained or fine-tuned BERT model, the training dataset (or a subset if it is too large) and the sentence or text to be explained.  Future work can extend this implementation to support RNNs and LSTMs. 

## Introspective Rationale Explainer
The IntrospectiveRationaleExplainer uses a generator-predictor framework to produce a comprehensive subset of text input features or rationales that are relevant for the classification task. This introspective model predicts the labels and incorporates the outcome into the rationale selection process. The outcome is a hard or soft selection of rationales (words that have useful information for the classification task) and anti-rationales (words that do not appear to have useful information). 

This [notebook](notebooks/text_classification/text_classification_introspective_rationale_explainer.ipynb) provides an example of how to use the introspective rationale generator.

### Preprocessing:
The IntrospectiveRationaleExplainer has generator and predictor modules that handle the required text pre-processing.

### Supported Models: 
The IntrospectiveRationaleExplainer is designed to be modular and extensible. The API currently has support for `RNN` and `BERT` models. There are three different sets of modules that has been implemented in this API:
* Explain a BERT model (BERT is used in the generator and predictor modules), 
* Explain an RNN model (RNNs are used in the generator and predictor modules), and
* Explain an RNN model with BERT as the generator (RNNs are used in the predictor module and BERT is used in the generator module)
The user can also explain a custom model. In this case, the user will have to provide the pre-processor, predictor and generator modulules to the API.  


<a  name="use"></a>
# Use Interpret-Text

## Interpretability in training

1. Train your model in a Jupyter notebook running on your local machine: For sample code on pre-processing and training, see [nlp-recipes](https://github.com/microsoft/nlp-recipes/blob/master/examples/text_classification/tc_mnli_transformers.ipynb) or our [sample notebook](notebooks/text_classification/text_classification_unified_information_explainer.ipynb).

2. Call the explainer: To initialize the explainers, you will need to pass either:

* the dataset or
* your own model, dataset, and other information depending on your choice of explainer

To initialize the `UnifiedInformationExplainer`, pass the model, the dataset you used to train the model along with the CUDA device and the BERT target layer.

``` python
from interpret_text.unified_information import UnifiedInformationExplainer

interpreter_unified = UnifiedInformationExplainer(model, 
                                 train_dataset, 
                                 device, 
                                 target_layer)
```

    If you intend to use the `ClassicalTextExplainer` with our default Linear Regression model, you can simply call the fit function with your dataset.
```python
from sklearn.preprocessing import LabelEncoder
from interpret_text.classical import ClassicalTextExplainer

explainer = ClassicalTextExplainer()
label_encoder = LabelEncoder()
classifier, best_params = explainer.fit(X_train, y_train)
```
    Instead, if you want to use the `ClassicalTextExplainer` with your own sklearn model, you will need to initialize `ClassicalTextExplainer` with your model, preprocessor and the range of hyperparamaters.
```python
from sklearn.preprocessing import LabelEncoder
from interpret_text.classical import ClassicalTextExplainer
from interpret_text.common.utils_classical import get_important_words, BOWEncoder

HYPERPARAM_RANGE = {
    "solver": ["saga"],
    "multi_class": ["multinomial"],
    "C": [10 ** 4],
}
preprocessor = BOWEncoder()
explainer = ClassicalTextExplainer(preprocessor, model, HYPERPARAM_RANGE)
```
## Instance-level (local) feature importance values
Get the local feature importance values: use the following function calls to explain an individual instance or a group of instances. 

```python
# explain the first data point in the test set
local_explanation = explainer.explain_local(x_test[0])

# sorted feature importance values and feature names
sorted_local_importance_names = local_explanation.get_ranked_local_names()
sorted_local_importance_values = local_explanation.get_ranked_local_values()
```

## Visualization Dashboard

### Initializing the `ExplanationDashboard` object

1. To use the visualization dashboard, import the `ExplanationDashboard` object from the package.

    ```python
    from interpret_text.widget import ExplanationDashboard
    ```
2. When initializing the ExplanationDashboard, you need to pass the local explanation object that is returned by our explainer.

    ```python
    ExplanationDashboard(local_explanantion)
    ```
    Note: if you are not using one of our explainers, you need to create your own explanation object by passing the feature importance values
    ```python
    from interpret_text.explanation.explanation import _create_local_explanation
    
    local_explanantion = _create_local_explanation(
    classification=True,
    text_explanation=True,
    local_importance_values=feature_importance_values,
    method=name_of_model,
    model_task="classification",
    features=parsed_sentence_list,
    classes=list_of_classes,
    )
    ```
### Using the Dashboard 
The dashboard visualizes the local feature importances of the document with an interactive bar chart and text area with highlighting and underlining of important words in your document. Words associated with positive feature importance contributed to the classification of the document towards the label indicated on the dashboard, words associated with negative feature importance contributed against it. The cap on number of important words is decided by the total number words with non-zero feature importances. Hovering over either the bars in the chart or the highlighted/underlined words will reveal a tooltip with the numerical feature importance. In the chart tooltip, the context of the word shows both the word before and after to allow users a way to differentiate between the same words used multiple times.

![Visualization Dashboard](/img/Interpret-text%20viz%20dashboard.gif)

<a  name="contrib"></a>

# Contributing
We welcome contributions and suggestions! Most contributions require you to agree to the Github Developer Certificate of Origin, DCO. For details, please visit  [https://probot.github.io/apps/dco/](https://probot.github.io/apps/dco/).

The Developer Certificate of Origin (DCO) is a lightweight way for contributors to certify that they wrote or otherwise have the right to submit the code they are contributing to the project. Here is the full text of the DCO, reformatted for readability:
```
By making a contribution to this project, I certify that:
(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
Contributors sign-off that they adhere to these requirements by adding a Signed-off-by line to commit messages.
This is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
Git even has a -s command line option to append this automatically to your commit message:
$ git commit -s -m 'This is my commit message'
```
When you submit a pull request, a DCO bot will automatically determine whether you need to certify. Simply follow the instructions provided by the bot.

<a name=code></a>
# Code of Conduct

This project has adopted the his project has adopted the  [GitHub Community Guidelines](https://help.github.com/en/github/site-policy/github-community-guidelines).

## Reporting Security Issues

Security issues and bugs should be reported privately, via email, to the Microsoft Security Response Center (MSRC) at  [secure@microsoft.com](mailto:secure@microsoft.com). You should receive a response within 24 hours. If for some reason you do not, please follow up via email to ensure we received your original message. Further information, including the  [MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155)  key, can be found in the  [Security TechCenter](https://technet.microsoft.com/en-us/security/default).
