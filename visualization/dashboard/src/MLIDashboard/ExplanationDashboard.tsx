import * as React from 'react'
import { IExplanationDashboardProps } from './Interfaces/IExplanationDashboardProps'
import { TextHighlighting } from './Control/TextHightlighting'
import { localization } from '../Localization/localization'
import { Slider } from 'office-ui-fabric-react/lib/Slider'
import _ from "lodash";
import { thisTypeAnnotation } from '@babel/types'

export interface IDashboardContext {
}

export interface IDashboardState {
}

export class ExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, IDashboardState> {
  state = {
    maxK: this.count_nonzeros(this.props.dataSummary.localExplanations),
    topK: this.count_nonzeros(this.props.dataSummary.localExplanations) / 2
  }
  public render () {
    return (
      <>
        <h1>{localization.interpretibilityDashboard}</h1>
        <div className = "explainerDashboard">
          <div className = "slidingBar">
              <Slider
                label={this.state.topK.toString().concat(" ",localization.importantWords)}
                min={1}
                max={this.state.maxK}
                step={1}
                defaultValue={(this.state.maxK / 2)}
                showValue={true}
                onChange={(value)=>this.setTopK(value)}
              />
            </div>
          <div className = "textHighlight">
            <TextHighlighting
              text = {this.props.dataSummary.text}
              localExplanations = {this.props.dataSummary.localExplanations}
              classNames = {this.props.dataSummary.classNames}
              prediction = {this.props.dataSummary.prediction}
              topK = {this.state.topK}
            />
          </div>
        </div>
      </>
    )
  }
  private setTopK(newNumber: number):void{
    this.setState({topK:newNumber})
  }
  private count_nonzeros(numArr: number[]):number{
    let counter = 0
    for (let i in numArr){
      if (numArr[i] == 0){
        counter++
      }
    }
    return counter
  }
}
