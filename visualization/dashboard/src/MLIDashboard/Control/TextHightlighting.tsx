import React from 'react'
import { IDatasetSummary } from '../Interfaces/IExplanationDashboardProps'
// import StackUtils from 'stack-utils'

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
    const importance = this.props.localExplanations
    const k = this.props.topK
    const sortedList = this.argsort(importance.map(Math.abs)).reverse().splice(0,k)
    return text.map((word, wordIndex) => {
      let styleType:any
      const score = importance[wordIndex]
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
  public argsort(toSort: number[]): number[] {
    const sorted = toSort.map((val, index) => [val, index]);
      sorted.sort((a, b) => {
          return a[0] - b[0];
      });
      return sorted.map(val => val[1]);
     }
}
