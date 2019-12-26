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
    const [data, x, y] = [[], [], []]
    sortedList.map(idx => {
      y.push(props.text[idx])
      x.push(importances[idx])
    })
    data.push({
      hoverinfo: 'text',
      orientation: 'h',
      type: 'bar',
      x,
      y
    })
    const chart = {
      data: data,
      layout: {
        title: localization.topFeatureList,
        xaxis:{range:[-1,1]}
      }
    }
    return chart
  }
}
