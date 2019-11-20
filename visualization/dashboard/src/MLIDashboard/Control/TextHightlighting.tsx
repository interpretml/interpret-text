import React from 'react'
import { IDatasetSummary } from '../Interfaces/IExplanationDashboardProps'

const highlighted = {
  color: 'white',
  backgroundColor: 'black',
  fontfamily: "segoe ui"
} as React.CSSProperties;

const boldunderline = {
  fontWeight: 'bold',
  textDecorationLine: 'underline',
  fontfamily: "segoe ui"
} as React.CSSProperties;

export class TextHighlighting extends React.PureComponent<IDatasetSummary> {

  public render (): React.ReactNode[] {
    const text = this.props.text;
    const importance = this.props.localExplanations;
    return text.map((word, wordIndex) => {
      let score = importance[wordIndex];
      if (score > 0) {
        return <span style={highlighted} title={score.toString()}>{word + ' '}</span>
      } else if (score < 0) {
        return <span style={boldunderline} title={score.toString()}>{word + ' '}</span>
      } else {
        return <span>{word + ' '}</span>
      }
    })
  }
}
