import * as React from 'react'
import { IExplanationDashboardProps } from './Interfaces/IExplanationDashboardProps'
import { TextHighlighting } from './Control/TextHightlighting'
import { localization } from '../Localization/localization'
import { Slider } from 'office-ui-fabric-react/lib/Slider'
import { BarChart } from './Control/BarChart'
import { Toggle } from 'office-ui-fabric-react/lib/Toggle'
import { Text } from 'office-ui-fabric-react/lib/Text'
import { RadioKeys, Utils } from './CommonUtils'
import { ChoiceGroup, IChoiceGroupOption } from 'office-ui-fabric-react/lib/ChoiceGroup'
import { FontWeights } from 'office-ui-fabric-react/lib/Styling'

const s = require('./ExplanationDashboard.css')

export interface IDashboardContext {}

export interface IDashboardState {
  maxK: number,
  topK: number,
  radio: string
}

const options: IChoiceGroupOption[] = [
  { key: RadioKeys.all, text: localization.allButton },
  { key: RadioKeys.pos, text: localization.posButton },
  { key: RadioKeys.neg, text: localization.negButton }
]

export class ExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, IDashboardState> {
  constructor(props: IExplanationDashboardProps, IDashboardState) {
    super(props)
    this.state = {
      maxK: Math.min(15, Math.ceil(this.countNonzeros(this.props.dataSummary.localExplanations))),
      topK: Math.ceil(this.countNonzeros(this.props.dataSummary.localExplanations) / 2),
      radio: RadioKeys.all
    }
    this.changeRadioButton = this.changeRadioButton.bind(this)
  }

  public render() {
    return (
      <div className='explainerDashboard'>
        <div className = 'sliderWithText' >
          <div className = 'slider'>
            <Slider
              min={1}
              max={this.state.maxK}
              step={1}
              defaultValue={(this.state.topK)}
              showValue={false}
              onChange={(value) => this.setTopK(value)}
            />
          </div>
          <div className = 'textBelowSlider'>
            {this.state.topK.toString() + ' ' + localization.importantWords}
          </div>
        </div>
        <div className = 'chartWithRadio'>
          <div className = 'barChart'>
            <BarChart
              text={this.props.dataSummary.text}
              localExplanations={this.props.dataSummary.localExplanations}
              topK={this.state.topK}
              radio={this.state.radio}
            />
          </div>
            <div className = 'chartRight'>
              <div className = 'labelPrediction' >
                {localization.label + localization.colon + this.predictClass(this.props.dataSummary.classNames, this.props.dataSummary.prediction)}
              </div>
              <div className = 'radio'>
                <ChoiceGroup defaultSelectedKey='all' options={options} onChange={this.changeRadioButton} required={true} />
              </div>
              <div className = 'chartLegend'>
                {localization.legendText}
              </div>
            </div>
        </div>
        <div className = 'highlightWithLegend'>
          <div className = 'textHighlighting'>
              <TextHighlighting
                text={this.props.dataSummary.text}
                localExplanations={this.props.dataSummary.localExplanations}
                topK={this.state.topK}
                radio={this.state.radio}
              />
          </div>
            <div className = 'textRight'>
              <div className = 'legend'>
                {localization.featureLegend}
              </div>
              <div>
    <span className = 'posFeatureImportance'>A</span>
              <span>{localization.posFeatureImportance}</span>
              </div>
              <div>
                <span className = 'negFeatureImportance'>A</span>
                <span> {localization.negFeatureImportance}</span>
              </div>
            </div>
        </div>
        </div>
    )
}
  private setTopK(newNumber: number): void {
    this.setState({ topK: newNumber })
  }

  private countNonzeros(numArr: number[]): number {
    let counter = 0
    for (const i in numArr) {
      if (numArr[i] !== 0) {
        counter++
      }
    }
    return counter
  }

  public predictClass(classname, prediction):string{
    return classname[Utils.argsort(prediction)[0]]
  }

  public changeRadioButton(ev: React.FormEvent<HTMLInputElement>, option: IChoiceGroupOption): void {
    this.setState({ radio: option.key })
  }
}
