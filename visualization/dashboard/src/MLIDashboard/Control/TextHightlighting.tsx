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
    const sortedList = this.argsort(importance).splice(0, k)
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

  public argsort (toSort:any[], direction?:string) {
    const sortedList = []
    for (const i in toSort) {
      sortedList.push([toSort[i], parseInt(i)])
    }
    if (typeof direction === 'undefined') {
      sortedList.sort(function (left, right) {
        return Math.abs(left[0]) > Math.abs(right[0]) ? -1 : 1
      })
    } else if (direction === 'positive') {
      sortedList.sort(function (left, right) {
        return left[0] > right[0] ? -1 : 1
      })
    } else if (direction === 'negative') {
      sortedList.sort(function (left, right) {
        return left[0] < right[0] ? -1 : 1
      })
    }
    const returnList = []
    for (const i in sortedList) {
      returnList.push(sortedList[i][1])
    }
    return returnList
  }
}
