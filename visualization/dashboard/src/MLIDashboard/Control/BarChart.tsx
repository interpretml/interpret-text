import React from 'react'
import { IChartProps } from '../Interfaces/IChartProps'
import { AccessibleChart, IPlotlyProperty, IData, PlotlyMode } from 'mlchartlib'
import { Utils } from '../CommonUtils'
import { localization } from '../../Localization/localization'

export class BarChart extends React.PureComponent<IChartProps> {
  /*
    * returns an accessible bar chart from mlchartlib
  */
  public render(): React.ReactNode {
    return (
      <AccessibleChart
        plotlyProps= {this.buildPlotlyProps(this.props)}
        sharedSelectionContext= {undefined}
        theme= {undefined}
      />
    )
  }

  private buildPlotlyProps(props): IPlotlyProperty {
    /* 
      * builds the bar chart with x and y values as well as the tooltip
      * defines the layout of the chart 
    */
    const importances = props.localExplanations
    const k = props.topK
    let sortedList = Utils.sortedTopK(importances, k, this.props.radio)
    const color = sortedList.map(x => importances[x] < 0 ? '#FFFFFF':'#5A53FF')
    const [data, x, y, ylabel, tooltip] = [[], [], [], [], []]
    sortedList.map((idx, i) => {
      let str = ''
      if (idx > 1) {
        str += '...'
      }
      if (idx > 0) {
        str += this.props.text[idx - 1] + ' '
      }
      str += this.props.text[idx]
      if (idx < this.props.text.length - 1) {
        str += ' ' + this.props.text[idx + 1]
      }
      if (idx < this.props.text.length - 2) {
        str += '...'
      }
      y.push(i)
      x.push(importances[idx])
      ylabel.push(this.props.text[idx])
      tooltip.push(str)
    })
    data.push({
      hoverinfo: 'x+text',
      orientation: 'h',
      text: tooltip,
      type: 'bar',
      marker: {
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
      config: { displaylogo: false, responsive: true,
        modeBarButtonsToRemove: ['toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian', 'lasso2d', 'select2d', 'sendDataToCloud', 'toImage', 'resetScale2d', 'autoScale2d', 'zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d']
      },
      data: data,
      layout: {
        xaxis: {
          range: [Math.floor(Math.min(...importances)) - 1, Math.ceil(Math.max(...importances)) + 1],
          fixedrange: true,
          title: localization.featureImportance,
          titlefont: {
            family: 'Segoe UI'
          },
          automargin:true
        },
        yaxis: {
          fixedrange: true,
          autotick: false,
          tickmode: 'array',
          tickvals: y,
          ticktext: ylabel,
          ticks: 'outside',
          ticklen: 8,
          tickwidth: 1,
          tickcolor: '#F2F2F2',
          titlefont: {
            family: 'Segoe UI'
          },
          automargin: true
        },
        margin: {
          t: 10,
          r: 0,
        },
        paper_bgcolor: '#f2f2f2',
        plot_bgcolor: '#FFFFFF'
      }
    }
    return chart
  }
}
