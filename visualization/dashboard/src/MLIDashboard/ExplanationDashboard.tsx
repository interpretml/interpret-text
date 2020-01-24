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
  { key: 'all', text: localization.allButton },
  { key: 'pos', text: localization.posButton },
  { key: 'neg', text: localization.negButton}
]

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
<<<<<<< HEAD
      radio: "all"
=======
      radio: RadioKeys.all
>>>>>>> dfb4a24f8e66103b7ebb1dc633066e8d4311ca77
    }
    this.changeRadioButton = this.changeRadioButton.bind(this)
  }

  public render() {
    return (
<<<<<<< HEAD
      <div style={{backgroundColor: "rgb(220,220,220)", fontFamily: 'Segoe UI'}}>
         <div className="explainerDashboard">
          <div className="ms-Grid" dir="ltr">
            <div className="ms-Grid-row">
              <div className="ms-Grid-col ms-sm6 ms-md6 ms-lg6">
                <div style={{marginRight: '5px'}}>
                  <div style={{marginTop: '5%'}}>
                    <Slider
                      min={1}
                      max={this.state.maxK}
                      step={1}
                      defaultValue={(this.state.topK)}
                      showValue={false}
                      onChange={(value) => this.setTopK(value)}
                    />
                  </div>
                  <div style={{fontSize: '1.8em', textAlign: 'center', fontStyle: 'italic', fontFamily: 'Segoe UI', color: '#636363', margin: '30px'}}>{this.state.topK.toString() + ' ' + localization.importantWords}</div>
                  <div style={{marginLeft:'5%', height: '350px'}}>
                   <BarChart
                      text={this.props.dataSummary.text}
                      localExplanations={this.props.dataSummary.localExplanations}
                      topK={this.state.topK}
                      radio={this.state.radio}
                    />
                    </div>
                    <div style={{fontFamily: 'Segoe UI', fontSize: '2.0em', fontWeight: "bold", marginBottom:'20px'}}>{localization.label + localization.colon + this.predictClass(this.props.dataSummary.classNames, this.props.dataSummary.prediction)}</div>
                    <div className="ms-Grid-col ms-sm6 ms-md6 ms-lg6" >
                      <ChoiceGroup defaultSelectedKey="all" options={options} onChange={this.changeRadioButton} required={true} />
                    </div>
                    <div className="ms-Grid-col ms-sm6 ms-md6 ms-lg6" >
                          {localization.legendText}
                        </div>
                </div>
              </div>
              <div className="ms-Grid-col ms-sm6 ms-md6 ms-lg6" style={{padding:'20px',}} >
                <div style={{marginLeft: '5px',}}>
                  <div style={{borderTopStyle: "solid", borderColor: 'rgb(0,120,212)', backgroundColor: 'white'}} >
                    <div style={{backgroundColor: 'white', borderStyle: 'groove', borderBlockColor: 'black', borderRadius: '5px', height: '250px', padding:'20px', overflowY: 'auto'}}>
                      <TextHighlighting
                        text={this.props.dataSummary.text}
                        localExplanations={this.props.dataSummary.localExplanations}
                        topK={this.state.topK}
                        radio={this.state.radio}
                      />
                  </div>
                  <div style={{margin: '30px 10px 10px 10px'}}>
                    {localization.featureLegend}
                  </div>
                  <span style={{color: 'white', fontWeight: 'bold', backgroundColor: '#0078D4', margin: '10px'}}>{localization.posFeatureImportance}</span>
                  <br></br>
                  <span style={{margin: '10px', fontWeight: 'bold', textDecorationLine: 'underline', color: '#0078D4'}}>{localization.negFeatureImportance}</span>                   
                  </div>
                </div>
=======
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
>>>>>>> dfb4a24f8e66103b7ebb1dc633066e8d4311ca77
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
<<<<<<< HEAD
=======

>>>>>>> dfb4a24f8e66103b7ebb1dc633066e8d4311ca77
  private countNonzeros(numArr: number[]): number {
    let counter = 0
    for (const i in numArr) {
      if (numArr[i] !== 0) {
        counter++
      }
    }
    return counter
  }
<<<<<<< HEAD
  public predictClass(classname, prediciton):string{
    return classname[Utils.argsort(prediciton)[0]]
=======

  public predictClass(classname, prediction):string{
    return classname[Utils.argsort(prediction)[0]]
  }

  public changeRadioButton(ev: React.FormEvent<HTMLInputElement>, option: IChoiceGroupOption): void {
    this.setState({ radio: option.key })
>>>>>>> dfb4a24f8e66103b7ebb1dc633066e8d4311ca77
  }
  public changeRadioButton(ev: React.FormEvent<HTMLInputElement>, option: IChoiceGroupOption): void {
    this.setState({radio: option.key})
  }
}
