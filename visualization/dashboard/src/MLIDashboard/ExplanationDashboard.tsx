import * as React from 'react'
import { IExplanationDashboardProps } from './Interfaces/IExplanationDashboardProps'
import { TextHighlighting } from './Control/TextHightlighting'

export interface IDashboardContext {
}

export interface IDashboardState {
}

export class ExplanationDashboard extends React.Component<IExplanationDashboardProps, IDashboardState> {
  public render () {
    return ( // look at how they do dataExploration
      <>
        <h1>Interpretability Dashboard</h1>
        {/* <div className="explainerDashboard">
                        Placeholder for the text explanation dashboard
        </div> */}
        <TextHighlighting
          text = {this.props.dataSummary.text}
          localExplanations = {this.props.dataSummary.localExplanations}
          classNames = {this.props.dataSummary.classNames}
          prediction = {this.props.dataSummary.prediction}
        />
      </>
    )
  }
}
