# Visualization Developer README

The visualization dashboard is a modular component that will take in an Explanation Object from any Explainer and visualize the results. This folder contains the Typescript code that will later be rendered into a Jupyter Widget.

# Contents
- [Getting Started](#getting-started)
- [Build](#build)
- [Widget](#widget)

<a name="getting-started"></a>
## Getting Started
### Setting up your environment
The dashboard uses Typescript and React as it's main language and framework. NPM is the package management system that will be used throughout this document, but other package management systems can be used.
### Feature Development
The dashboard current uses [Office UI Fabric](https://developer.microsoft.com/en-us/fabric#/get-started) and [mlchartlib](https://github.com/interpretml/interpret-community/tree/master/visualization/mlchartlib) for development.

The source code is broken into localization and MLIDashboard. Localization is used as to display the static strings in the dashboard to different languages, while MLIDashboard houses the Typescript code for functionality.

The dashboard is then broken down into controls and interfaces. The control folder contains the components used in the dashboard while the interfaces provides the skeleton structure for the data needed to be passed into the component. 

### Styling
The dashboard uses [SCSS](https://sass-lang.com/) for styling. [Flexbox](https://css-tricks.com/snippets/css/a-guide-to-flexbox/) is used to setup the layout of the dashboard. 

<a name="build"></a>
## Build
Before testing in the widget, the dashboard can be tested locally by running the following commands.

- from interpret-community-text/visualization/dashboard
	> npm run-script build
	> npm run-script build-css
- from from interpret-community-text/visualization/test
	> npm run-script build
	> npm start

This will open up a local web app which will display the dashboard. Then, the dashboard can be converted into an ipywidget and be rendered in a Jupyter Notebook.

<a name="widget"></a>
## Widget
The README in the widget folder describes the contract between the Python API and Explanation Dashboard. It also details how to build and compile the dashboard in widget form. 