import React from 'react'
import { IDatasetSummary } from '../Interfaces/IExplanationDashboardProps'

const highlighted = {
  color: 'white',
  backgroundColor: 'black',
  fontfamily: 'segoe ui'
} as React.CSSProperties

const boldunderline = {
  fontWeight: 'bold',
  textDecorationLine: 'underline',
  fontfamily: 'segoe ui'
} as React.CSSProperties

export class TextHighlighting extends React.PureComponent<IDatasetSummary> {
  public render (): React.ReactNode[] {
    const text = this.props.text
    const importance = this.props.localExplanations
    const k = this.props.topK
    const sortedList = this.argsort(importance).splice(0,k)
    return text.map((word, wordIndex) => {
      if (sortedList.includes(wordIndex)) {
        const score = importance[wordIndex]
        if (score > 0) {
          return <span style = {highlighted} title={score.toString()}>{word + " "}</span>
        } else if (score < 0) {
          return <span style = {boldunderline} title={score.toString()}>{word + " "}</span>
        } else {
          return <span>{word + " "}</span>
        }
      }
      else {
        return <span>{word + " "}</span>
      }
    })
  }

  public argsort(toSort:any[], direction?:string){
    let sortedList = []
    for (let i in toSort){
      sortedList.push([toSort[i], parseInt(i)])
    }
    if (typeof direction == "undefined"){
      sortedList.sort(function(left, right){
        return Math.abs(left[0]) > Math.abs(right[0]) ? -1:1
      })
    }
    else if (direction == "positive"){
      sortedList.sort(function(left, right){
        return left[0] > right[0] ? -1:1
      })
    }
    else if (direction == "negative"){
      sortedList.sort(function(left, right){
        return left[0] < right[0] ? -1:1
      })
    }
    let returnList = []
    for (let i in sortedList){
      returnList.push(sortedList[i][1])
    }
    return returnList
  }
}
