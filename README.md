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

<details>

<summary><strong><em>1. Clone the interpret-community repository</em></strong></summary>

Clone and cd into the repository
```
git clone https://github.com/microsoft/interpret-community-text.git
cd interpret-community-text
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
<br></br>
</details>

<details>
<summary><strong><em>3. Install package </em></strong></summary>

You can install the package from source or from pipy.

<details><summary><strong><em>3.1 From source (developers): </strong></em></summary>

```
    pip install -e .
```
</details>

<details><summary><strong><em>3.2 From pipy (package users): </strong></em></summary>

```
    pip install keyring artifacts-keyring
    pip install interpret-text --index-url "https://pkgs.dev.azure.com/responsibleai/_packaging/responsibleai/pypi/simple" (placehodler)
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


To set up your visualization dashboard:

```
TODO
```

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

<a name="code-of-conduct"></a>
# Code of Conduct

<a name=Refs></a>
# Additional References
