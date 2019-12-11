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
    topK:this.props.config.topK,
  }
  public render () {
    return (
      <>
        <h1>{localization.interpretibilityDashboard}</h1>
        <div className = "explainerDashboard">
          <div className = "slidingBar">
              <Slider
                label={localization.topKwords}
                min={0}
                max={this.props.dataSummary.text.length}
                step={1}
                defaultValue={this.props.config.topK}
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
}
