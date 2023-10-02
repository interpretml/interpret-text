# Widget Developer README

The widget holds the files needed to run the visualization dashboard in a Jupyter Notebook

# Contents
- [Getting Started](#getting-started)
- [Build](#build)

<a name="getting-started"></a>
## Getting Started
Once the Typescript code has been built using the dashboard build instructions, it can be moved into an ipywidget to be rendered in a Jupyter notebook.

### Understanding the contracts
The widget contract is defined between 
[ExplanationWidget.py](https://github.com/interpretml/interpret-text/blob/main/python/interpret_text/experimental/widget/ExplanationWidget.py) and the [explanationDashboard.tsx](https://github.com/interpretml/interpret-text/blob/main/python/interpret_text/experimental/widget/js/src/explanationDashboard.tsx) (in widget/js/src). These two files must be in sync for the dashboard to instantiate.

Additionally, the  [ExplanationDashboard.py](https://github.com/interpretml/interpret-text/blob/main/python/interpret_text/experimental/widget/ExplanationDashboard.py) takes in an Explanation Object which is used to initialize the dashboard. 



<a name="build"></a>
## Build
Run the following commands 

- From interpret-text\python\interpret_text\widget\js
	> npm install
	> npm run-script build:all

- Manually delete the folder called node_modules from the current folder

- From interpret-text\python
	> pip install .

After you delete the node_modules folder, the widget needs to be installed which is why you run the last command. Similarly, you can also update the pypi feed with the new files in the static folder so users can upgrade their interpret-text package with the most up-to-date dashboard.