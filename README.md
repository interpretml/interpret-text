# Contents

- [Overview of Interpret-Text](#overview)
- [Target Audience](#target-audience)
- [Getting Started](#getting-started)
- [Supported Models and NLP Scenarios](#models)
- [Supported Explainers](#explainers)
- [Use Interpret-Text](#use)
- [Contributing](#contrib)
- [Code of Conduct](#code-of-conduct)
- [Additional References](#Refs)

<a name="overview"></a>
# Overview of Interpret-Text
The Interpret Community Text builds on [Interpret](https://github.com/interpretml/interpret-community), an open source python package from Microsoft Research for training interpretable models and helping to explain blackbox systems. We have added extensions to interpret NLP models.

This repository contains an SDK and Jupyter notebooks with examples to showcase its use.

<a name="target-audience"></a>
# Target Audience

<a name="getting-started"></a>
# Getting Started

This repo uses Anaconda to simplify package and environment management.

To setup on your local machine:

<details><summary><strong><em>1. Set up Environment</em></strong></summary>

    a. Install Anaconda with Python >= 3.7 
       [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) is a quick way to get started.

 
    b. Create conda environment named interp_text

```
    conda create -n interp_text python=3.7
```

    Optional, additional reading:

    - [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
    - [jupyter](https://pypi.org/project/jupyter/)
    - [nb_conda](https://github.com/Anaconda-Platform/nb_conda)

<details><summary><strong><em>On Windows: c. Activate conda environment</strong></em></summary>

```
    conda activate interp_text
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
    conda activate interp_text
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
```
</details>

<!---{% from interpret.ext.blackbox import TabularExplainer %}
--->

To set up on Azure:

To set up your visualization dashboard:

# <a name="models"></a>
# Supported Models and NLP Scenarios


# <a name="explainers"></a>
# Supported Explainers
Explain what the baseline is, and MSRA/RNP as well as links to research/conferences

<a name="use"></a>
# Use Interpret-Text
Teaches the user how to use this package and links to sample notebooks. 

<a name="contrib"></a>
# Contributing
