import { DOMWidgetModel, DOMWidgetView } from '@jupyter-widgets/base';
import { ExplanationDashboard } from 'mlchartlib';
import * as _ from 'lodash';
import React from 'react';
import ReactDOM from 'react-dom';

// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including
//
//  - `_view_name`
//  - `_view_module`
//  - `_view_module_version`
//
//  - `_model_name`
//  - `_model_module`
//  - `_model_module_version`
//
//  when different from the base class.

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
export class  ExplanationModel extends DOMWidgetModel {
    defaults() {
        return {
            _model_name : 'ExplanationModel',
            _view_name : 'ExplanationView',
            _model_module : 'interpret-text-widget',
            _view_module : 'interpret-text-widget',
            _model_module_version : '0.1.0',
            _view_module_version : '0.1.1',
            value: {},
            request: {},
            response: {}
        }
    }
};

interface IPromiseResolvers {
}

// Custom View. Renders the widget model.
export class ExplanationView extends DOMWidgetView {
    el: any;
    public render() {
        this.el.style.cssText = "width: 100%";
        let root_element = document.createElement("div");
        root_element.style.cssText = "width: 100%;";
        ReactDOM.render(<ExplanationDashboard/>, root_element);
        this.el.appendChild(root_element)
    }
    
};
