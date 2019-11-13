import React from 'react'
import { IDatasetSummary } from '../Interfaces/IExplanationDashboardProps'

export class TextHighlighting extends React.Component<IDatasetSummary> {
  public render (): React.ReactNode[] {
    const text = this.props.text
    const importance = this.props.localExplanations
    return text.map((word, wordIndex) => {
      if (importance[wordIndex] > 0) {
        return <span className="highlighted" title={word}>{word + ' '}</span>
      } else if (importance[wordIndex] < 0) {
        return <span className="boldunderline" title={word}>{word + ' '}</span>
      } else {
        return <span>{word + ' '}</span>
      }
    })
  }
}
