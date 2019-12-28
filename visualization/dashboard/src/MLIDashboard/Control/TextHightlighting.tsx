import React from 'react'
import { IDatasetSummary } from '../Interfaces/IExplanationDashboardProps'
import { Utils } from '../CommonUtils'

const highlighted = {
  color: 'white',
  backgroundColor: '#0078D4',
  fontfamily: 'Segoe UI'
} as React.CSSProperties

const boldunderline = {
  color: '#0078D4',
  fontWeight: 'bold',
  textDecorationLine: 'underline',
  fontfamily: 'Segoe UI'
} as React.CSSProperties

export class TextHighlighting extends React.PureComponent<IDatasetSummary> {
  public render (): React.ReactNode[] {
    const text = this.props.text
    const importances = this.props.localExplanations
    const k = this.props.topK
    const sortedList = Utils.argsort(importances.map(Math.abs)).reverse().splice(0, k)
    return text.map((word, wordIndex) => {
      let styleType:any
      const score = importances[wordIndex]
      if (sortedList.includes(wordIndex)) {
        if (score > 0) {
          styleType = highlighted
        } else if (score < 0) {
          styleType = boldunderline
        }
      }
      return <span style = {styleType} title = {score.toString()}>{word + ' '}</span>
    })
  }
}
