
# Interpret Community Text SDK
The Interpret Community Text builds on [Interpret-Community](https://github.com/interpretml/interpret-community), an open source python package for training interpretable models and helping to explain blackbox machine learning systems. We have added extensions to support text models.

This repository contains an SDK and Jupyter notebooks with examples to showcase its use.

# Contents

  

-  [Overview of Interpret-Text](#overview)

-  [Target Audience](#target-audience)

-  [Getting Started](#getting-started)

-  [Supported Models and NLP Scenarios](#models)

-  [Supported Explainers](#explainers)

-  [Use Interpret-Text](#use)

-  [Contributing](#contrib)

-  [Code of Conduct](#code-of-conduct)
  

<a  name="overview"></a>

# Overview of Interpret-Text
Interpret-Text incorporates community developed interpretability techniques for NLP models and a visualization dashboard to view the results. Users can run their experiments across multiple state-of-the-art explainers and easily perform comparative analysis on them. Using these tools, users will be able to explain their machine learning models globally on each label, or locally for each document. In particular, this open-source toolkit:
1. Actively incorporates innovative text interpretability techniques, and allows the community to further expand it's offerings
2. Creates a common API across the integrated libraries
3. Provides an interactive visualization dashboard to empower its users to gain insights into their data

<a  name="target-audience"></a>

# Target Audience

1. Developers/Data Scientists: Having all of the interpretability techniques in one place makes it easy for data scientists to experiment with different interpretability techniques, and explain their model in a scalable and seamless manner. The set of rich interactive visualizations allow developers and data scientists to train and deploy more transparent machine learning models instead of wasting time and effort on generating customized visualizations, addressing scalability issues by optimizing third-party interpretability techniques, and adopting/operationalizing interpretability techniques.

  

2. Business Executives: The core logic and visualizations are beneficial for raising awareness among those involved in developing AI applications, allow them to audit model predictions for potential unfairness, and establish a strong governance framework around the use of AI applications.

  

3. Machine Learning Interpretability Researchers: Interpret's extension hooks make it easy to extend and thus, interpretability researchers who are interested in adding their own techniques, can easily add them to the community repository and compare it to state-of-the-art and proven interpretability techniques and/or other community techniques.

  

<a  name="getting-started"></a>

# Getting Started

  

This repository uses Anaconda to simplify package and environment management.

To setup on your local machine:

<details><summary><strong><em>1. Set up Environment</em></strong></summary>

a. Install Anaconda with Python = 3.7
  
b. Create conda environment named interpret_text

```

conda create -n interpret_text python=3.7

```

Optional, additional reading:

- [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

- [jupyter](https://pypi.org/project/jupyter/)

- [nb_conda](https://github.com/Anaconda-Platform/nb_conda)

  

<details><summary><strong><em>On Windows: c. Activate conda environment</strong></em></summary>

```

conda activate interpret_text

pip install pytest

pip install lightgbm

pip install interpret-community

pip install xgboost

pip install tensorflow

conda install pytorch torchvision cpuonly -c pytorch

```

</details>

<details><summary><strong><em>On Linux:</em> c. Activate conda environment</em></strong></summary>

```

conda activate interpret_text

```

</details>

<br></br>

</details>

<details>

<summary><strong><em>2. Clone the interpret-community repository</em></strong></summary>

Clone and cd into the repository

```

git clone https://github.com/microsoft/interpret-community-text.git

cd interpret-community-text

```

</details>

<details>

<summary><strong><em>3. Install Python module, packages and necessary distributions</em></strong></summary>

```

pip install -e ./python

pip install pytest

pip install lightgbm

pip install interpret-community

pip install xgboost

pip install tensorflow

conda install pytorch torchvision cpuonly -c pytorch

```

If you intend to run repo tests:

```

pip install pytest

pip install lightgbm

pip install interpret-community

pip install xgboost

pip install tensorflow

conda install pytorch torchvision cpuonly -c pytorch

```

</details>

<details>

<summary><strong><em>4. Set up and run Jupyter Notebook server </em></strong></summary>

  
Install and run Jupyter Notebook

```
conda install -c conda-forge notebook
jupyter notebook
```

</details>

<!---{% from interpret.ext.blackbox import TabularExplainer %}

--->

To set up on Azure:

To set up your visualization dashboard:

# <a name="models"></a>

# Supported Models and NLP Scenarios

 Currently this repository provides support for the the text classification scenario.
 
 The API supports models that are trained on datasets in Python's `scipy.sparse.csr_matrix` format.
  
  The explanation functions accept both models and pipelines as input, as long as the model or pipeline follows the sklearn's classifier API.

# <a name="explainers"></a>

# Supported Explainers
The following is a list of the explainers available in this repository:
* [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model)
* [MSR-Asia](https://www.microsoft.com/en-us/research/publication/towards-a-deep-and-unified-understanding-of-deep-neural-models-in-nlp/): uses an information-based measure to provide explanations on the intermediate layers of deep NLP models
  

<a  name="use"></a>

# Use Interpret-Text

Teaches the user how to use this package and links to sample notebooks.

  

<a  name="contrib"></a>

# Contributing
This project welcomes contributions and suggestions. Most contributions require you to agree to the Github Developer Certificate of Origin, DCO. For details, please visit  [https://probot.github.io/apps/dco/](https://probot.github.io/apps/dco/).

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

# Code of Conduct

This project has adopted the his project has adopted the  [GitHub Community Guidelines](https://help.github.com/en/github/site-policy/github-community-guidelines).

## Reporting Security Issues

Security issues and bugs should be reported privately, via email, to the Microsoft Security Response Center (MSRC) at  [secure@microsoft.com](mailto:secure@microsoft.com). You should receive a response within 24 hours. If for some reason you do not, please follow up via email to ensure we received your original message. Further information, including the  [MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155)  key, can be found in the  [Security TechCenter](https://technet.microsoft.com/en-us/security/default).