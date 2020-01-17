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
    let color: string[]
    const k = props.topK
    let sortedList = Utils.sortedTopK(importances, k, this.props.radio)
    // color = sortedList.map(x=>importances[x]<0?'rgb(255,255,255)':'rgb(0,120,212)')
    color = sortedList.map(x=>importances[x]<0?'#FFFFFF':'#5A53FF');
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
      config:{displaylogo: false, responsive: true, modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d', 
      'sendDataToCloud', 'toImage', 'resetScale2d', 'autoScale2d', 'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d']},
      data: data,
      layout: {
        xaxis:{
          title: localization.featureImportance
        }
      }
    }
    return chart
  }
}
