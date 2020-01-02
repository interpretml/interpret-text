import React from 'react'
import { IChartProps } from '../Interfaces/IChartProps'
import { AccessibleChart, IPlotlyProperty, IData, PlotlyMode } from 'mlchartlib'
import { Utils } from '../CommonUtils'
import { localization } from '../../Localization/localization'

export class BarChart extends React.PureComponent<IChartProps> {
  public render (): React.ReactNode {
    return (
      <AccessibleChart
        plotlyProps= {this.buildPlotlyProps(this.props)}
        sharedSelectionContext={undefined}
        theme = "light"
      />
    )
  }

  private buildPlotlyProps (props): IPlotlyProperty {
    const importances = props.localExplanations
    let sortedList: number[]
    let color: string[]
    const k = props.topK
    if ((this.props.posOnly && this.props.negOnly) || (!this.props.posOnly && !this.props.negOnly)){
      sortedList = Utils.argsort(importances.map(Math.abs)).reverse().splice(0, k).reverse()
    } 
    else if (this.props.negOnly){
      sortedList = Utils.argsort(importances).splice(0, k).reverse()
    }
    else {
      sortedList = Utils.argsort(importances).reverse().splice(0, k).reverse()
    }   
    color = sortedList.map(x=>importances[x]<0?'rgb(255,255,255)':'rgb(0,120,212)');
    console.log(importances);
    const [data, x, y] = [[], [], []]
    sortedList.map(idx => {
      y.push(props.text[idx])
      x.push(importances[idx])
    })
    data.push({
      hoverinfo: 'text',
      orientation: 'h',
      type: 'bar',
      marker:{
        color,
        line: {
          color: 'rgb(0,120,212)',
          width: 1.5
        }
      },
      x,
      y
    })
    const chart = {
      data: data,
      layout: {
        title: localization.featureImportance,
        xaxis:{range:[-1,1]}
      }
    }
    return chart
  }
}
