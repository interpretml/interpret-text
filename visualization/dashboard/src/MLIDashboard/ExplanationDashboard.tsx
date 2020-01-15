import * as React from 'react'
import {IExplanationDashboardProps} from './Interfaces/IExplanationDashboardProps'
import {TextHighlighting} from './Control/TextHightlighting'
import {localization} from '../Localization/localization'
import {Slider} from 'office-ui-fabric-react/lib/Slider'
import {BarChart} from './Control/BarChart'
import {Toggle} from 'office-ui-fabric-react/lib/Toggle';
import {Text} from 'office-ui-fabric-react/lib/Text';
import { Utils } from './CommonUtils'
import { ChoiceGroup, IChoiceGroupOption } from 'office-ui-fabric-react/lib/ChoiceGroup';
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

export class ExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, IDashboardState> {
  constructor(props: IExplanationDashboardProps, IDashboardState) {
    super(props)
    this.state = {
      maxK: this.countNonzeros(this.props.dataSummary.localExplanations),
      topK: Math.ceil(this.countNonzeros(this.props.dataSummary.localExplanations) / 2),
      radio: "all"
    }
    this.changeRadioButton = this.changeRadioButton.bind(this)
  }


  public render() {
    return (
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
              </div>
            </div>
          </div>
        </div>
      </div>
    )
}
  private setTopK(newNumber: number): void {
    this.setState({topK: newNumber})
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
  public predictClass(classname, prediciton):string{
    return classname[Utils.argsort(prediciton)[0]]
  }
  public changeRadioButton(ev: React.FormEvent<HTMLInputElement>, option: IChoiceGroupOption): void {
    this.setState({radio: option.key})
  }
}
