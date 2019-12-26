import React from 'react'
import { IDatasetSummary } from '../Interfaces/IExplanationDashboardProps'
import { AccessibleChart, IPlotlyProperty, IData, PlotlyMode } from 'mlchartlib'
import { Utils } from '../CommonUtils'
import { localization } from '../../Localization/localization'

export class BarChart extends React.PureComponent<IDatasetSummary> {
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
    const sortedList = Utils.argsort(importances.map(Math.abs)).reverse().splice(0, props.topK)
    const [data, x, y] = [[], [], []]
    sortedList.map(idx => {
      y.push(props.text[idx])
      x.push(importances[idx])
    })
    x.reverse()
    y.reverse()
    data.push({
      hoverinfo: 'text',
      orientation: 'h',
      type: 'bar',
      name: 'temp Name',
      x,
      y
    })
    const chart = {
      data: data,
      layout: {
        title: localization.topFeatureList
      }
    }
    return chart
  }
}
