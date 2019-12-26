import * as React from 'react'
import { IExplanationDashboardProps } from './Interfaces/IExplanationDashboardProps'
import { TextHighlighting } from './Control/TextHightlighting'
import { localization } from '../Localization/localization'
import { Slider } from 'office-ui-fabric-react/lib/Slider'
import { BarChart } from './Control/BarChart'
import { Toggle } from 'office-ui-fabric-react/lib/Toggle';

export interface IDashboardContext {
}

export interface IDashboardState {
  maxK: number,
  topK: number,
  posToggle: boolean,
}

export class ExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, IDashboardState> {
  constructor(props: IExplanationDashboardProps, IDashboardState){
    super(props)
    this.state = {
      maxK: this.countNonzeros(this.props.dataSummary.localExplanations),
      topK: Math.ceil(this.countNonzeros(this.props.dataSummary.localExplanations) / 2),
      posToggle: false
    }
    this.setPosToggle = this.setPosToggle.bind(this)
  }


  public render () {
    return (
      <>
        <h1>{localization.interpretibilityDashboard}</h1>
        <div className = "explainerDashboard">
          <Slider
            label={this.state.topK.toString().concat(' ', localization.importantWords)}
            min={1}
            max={this.state.maxK}
            step={1}
            defaultValue={(this.state.topK)}
            showValue={true}
            onChange={(value) => this.setTopK(value)}
          />
          <TextHighlighting
            text = {this.props.dataSummary.text}
            localExplanations = {this.props.dataSummary.localExplanations}
            topK = {this.state.topK}
          />
          <BarChart
            text = {this.props.dataSummary.text}
            localExplanations = {this.props.dataSummary.localExplanations}
            topK = {this.state.topK}
          />
          <Toggle label="With inline label and without onText and offText" inlineLabel onChange={this.setPosToggle} />

        </div>
      </>
    )
  }

  private setTopK (newNumber: number):void{
    this.setState({ topK: newNumber })
  }

  private countNonzeros (numArr: number[]):number {
    let counter = 0
    for (const i in numArr) {
      if (numArr[i] !== 0) {
        counter++
      }
    }
    return counter
  }
  public setPosToggle(ev: React.MouseEvent<HTMLElement>, checked: boolean) {
    //console.log('toggle is ' + (checked ? 'checked' : 'not checked'));
    this.setState({posToggle: !this.state.posToggle})
    console.log(this.state.posToggle)
  }
}
