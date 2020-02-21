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
[ExplanationWidget.py](https://github.com/microsoft/interpret-community-text/blob/master/python/interpret_text/widget/ExplanationWidget.py) and the [explanationDashboard.tsx](https://github.com/microsoft/interpret-community-text/blob/eac1b7a57f1d2c1b9e9fa22aca237e0cc97b454b/python/interpret_text/widget/js/src/explanationDashboard.tsx) (in widget/js/src). These two files must be in sync for the dashboard to instantiate.

Additionally, the  [ExplanationDashboard.py](https://github.com/microsoft/interpret-community-text/blob/master/python/interpret_text/widget/ExplanationDashboard.py) takes in an Explanation Object which is used to initialize the dashboard. 



<a name="build"></a>
## Build
Run the following commands 

- From interpret-community-text\python\interpret_text\widget\js
	> npm install
	> npm run-script build:all

- Manually delete the folder called node_modules from the current folder

- From interpret-community-text\python
	> pip install .

After you delete the node_modules folder, the widget needs to be installed which is why you run the last command. Similarly, you can also update the pypi feed with the new files in the static folder so users can upgrade their interpret-text package with the most up-to-date dashboard.