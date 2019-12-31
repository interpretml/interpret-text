import React from 'react'
import { IChartProps } from '../Interfaces/IChartProps'
import { Utils } from '../CommonUtils'

const highlighted = {
  color: 'white',
  backgroundColor: '#0078D4',
  fontFamily: 'Segoe UI',
  fontSize: '1.5em',
} as React.CSSProperties

const boldunderline = {
  color: '#0078D4',
  fontWeight: 'bold',
  textDecorationLine: 'underline',
  fontFamily: 'Segoe UI',
  fontSize: '1.5em',
} as React.CSSProperties

const normal = {
  fontFamily: 'Segoe UI',
  fontSize: '1.5em',
} as React.CSSProperties

export class TextHighlighting extends React.PureComponent<IChartProps> {
  public render (): React.ReactNode[] {
    const text = this.props.text
    const importances = this.props.localExplanations
    const k = this.props.topK
    let sortedList: number[]
    if ((this.props.posOnly && this.props.negOnly) || (!this.props.posOnly && !this.props.negOnly)){
      sortedList = Utils.argsort(importances.map(Math.abs)).reverse().splice(0, k)
    } 
    else if (this.props.negOnly){
      sortedList = Utils.argsort(importances).splice(0, k)
    }
    else {
      sortedList = Utils.argsort(importances).reverse().splice(0, k)
    }
    let val = text.map((word, wordIndex) => {
      let styleType = normal
      const score = importances[wordIndex]
      if (sortedList.includes(wordIndex)) {
        if (score > 0) {
          styleType = highlighted
        } else if (score < 0) {
          styleType = boldunderline
        } else {
          styleType = normal
        }
      }
      return ( <span style = {styleType} title = {score.toString()}>{word}</span>)
    })
    return val.map((word, index)=>{
      return <span>{word} </span>
    })
  }
}
