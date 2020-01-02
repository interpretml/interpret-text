import * as React from 'react'
import {IExplanationDashboardProps} from './Interfaces/IExplanationDashboardProps'
import {TextHighlighting} from './Control/TextHightlighting'
import {localization} from '../Localization/localization'
import {Slider} from 'office-ui-fabric-react/lib/Slider'
import {BarChart} from './Control/BarChart'
import {Toggle} from 'office-ui-fabric-react/lib/Toggle';
import {Text} from 'office-ui-fabric-react/lib/Text';
import { Utils } from './CommonUtils'
export interface IDashboardContext {}

export interface IDashboardState {
  maxK: number,
  topK: number,
  posToggle: boolean,
  negToggle: boolean
}

export class ExplanationDashboard extends React.PureComponent<IExplanationDashboardProps, IDashboardState> {
  constructor(props: IExplanationDashboardProps, IDashboardState) {
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


  public render() {
    return (
      <div style={{backgroundColor: "rgb(220,220,220)", fontFamily: 'Segoe UI'}}>
         <div className="explainerDashboard">
          <div className="ms-Grid" dir="ltr">
            <div className="ms-Grid-row">
              <div className="ms-Grid-col ms-sm6 ms-md6 ms-lg6">
                <div style={{marginRight: '5px'}}>
                  <div style={{fontFamily: 'Segoe UI', fontSize: '2.0em', fontWeight: "bold"}}>{localization.userText}</div>
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
                  <div style={{fontSize: '1.8em', textAlign: 'center', fontStyle: 'italic', fontFamily: 'Segoe UI', color: '#636363', margin: '30px'}}>{this.state.topK.toString() + ' ' + localization.importantWords)}</div>
                  <div style={{backgroundColor: 'white', borderStyle: 'groove', borderBlockColor: 'black', borderRadius: '5px', height: '600px', padding:'20px'}}>
                    <TextHighlighting
                      text={this.props.dataSummary.text}
                      localExplanations={this.props.dataSummary.localExplanations}
                      topK={this.state.topK}
                      posOnly={this.state.posToggle}
                      negOnly={this.state.negToggle}

                    />
                  </div>
                  <div style={{margin: '30px 10px 10px 10px'}}>
                    {localization.featureLegend}
                  </div>
                  <span style={{color: 'white', fontWeight: 'bold', backgroundColor: '#0078D4', margin: '10px'}}>{localization.posFeatureImportance}</span>
                  <br></br>
                  <span style={{margin: '10px', fontWeight: 'bold', textDecorationLine: 'underline'}}>{localization.negFeatureImportance}</span>
                </div>
              </div>
              <div className="ms-Grid-col ms-sm6 ms-md6 ms-lg6" style={{padding:'20px',}} >
                <div style={{marginLeft: '5px',}}>
                  <div style={{fontFamily: 'Segoe UI', fontSize: '2.0em', fontWeight: "bold", marginBottom:'20px'}}>Label: Spam</div>
                  <div style={{borderTopStyle: "solid", borderColor: 'rgb(0,120,212)', backgroundColor: 'white'}} >
                    <div style={{marginLeft: '5%', fontFamily: 'Segoe UI', fontSize: '2.5em', marginTop:'20px'}}>{localization.topFeatureList}</div>
                    <div style={{marginLeft: '5%', fontFamily: 'Segoe UI', fontSize: '1.8em',}}>{localization.model + localization.colon + this.props.modelInformation.model}</div>
                   <div style={{marginLeft:'5%'}}>
                   <BarChart
                      text={this.props.dataSummary.text}
                      localExplanations={this.props.dataSummary.localExplanations}
                      topK={this.state.topK}
                      posOnly={this.state.posToggle}
                      negOnly={this.state.negToggle}
                    />
                    </div>
                    <div className="ms-Grid" dir="ltr" >
                      <div className="ms-Grid-row" style={{margin: '5%'}}>
                        <div className="ms-Grid-col ms-sm6 ms-md6 ms-lg6" >
                          <Toggle label={localization.posToggle} inlineLabel onChange={this.setPosToggle} />
                          <Toggle label={localization.negToggle} inlineLabel onChange={this.setNegToggle} />
                        </div>
                        <div className="ms-Grid-col ms-sm6 ms-md6 ms-lg6" >
                          {localization.legendText}
                        </div>
                      </div>
                    </div>
                    
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
  public setPosToggle(ev: React.MouseEvent<HTMLElement>, checked: boolean) {
    this.setState({posToggle: !this.state.posToggle})
    //this.setState({negToggle: false})
  }
  public setNegToggle() {
    this.setState({negToggle: !this.state.negToggle})
    //this.setState({posToggle: false})
  }
}
