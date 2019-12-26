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
  negToggle: boolean
}

export class ExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, IDashboardState> {
  constructor(props: IExplanationDashboardProps, IDashboardState){
    super(props)
    this.state = {
      maxK: this.countNonzeros(this.props.dataSummary.localExplanations),
      topK: Math.ceil(this.countNonzeros(this.props.dataSummary.localExplanations) / 2),
      posToggle: false,
      negToggle: false
    }
    this.setPosToggle = this.setPosToggle.bind(this)
    this.setNegToggle = this.setNegToggle.bind(this)
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
            posOnly = {this.state.posToggle}
            negOnly = {this.state.negToggle}
          />
          <BarChart
            text = {this.props.dataSummary.text}
            localExplanations = {this.props.dataSummary.localExplanations}
            topK = {this.state.topK}
            posOnly = {this.state.posToggle}
            negOnly = {this.state.negToggle}
          />
          <Toggle label={localization.posToggle} inlineLabel onChange={this.setPosToggle} />
          <Toggle label={localization.negToggle} inlineLabel onChange={this.setNegToggle} />
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
    this.setState({posToggle: !this.state.posToggle})
    //this.setState({negToggle: false})
  }
  public setNegToggle() {
    this.setState({negToggle: !this.state.negToggle})
    //this.setState({posToggle: false})
  }
}
