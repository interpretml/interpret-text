import * as React from 'react'
import { IExplanationDashboardProps } from './Interfaces/IExplanationDashboardProps'
import { TextHighlighting } from './Control/TextHightlighting'
import { localization } from '../Localization/localization'

export interface IDashboardContext {
}

export interface IDashboardState {
}

export class ExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, IDashboardState> {
  public render () {
    return (
      <>
        <h1>{localization.interpretibilityDashboard}</h1>
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
