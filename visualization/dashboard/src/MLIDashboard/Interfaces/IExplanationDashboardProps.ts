export interface IExplanationDashboardProps {
  /*
    * the interface design for the dashboard
  */
  modelInformation: IModelInformation;
  dataSummary: IDatasetSummary;
  }

export interface IModelInformation {
  /*
    * information about the model used
  */
  model: 'msra'| 'bow' | 'rmp';
}

export interface IDatasetSummary {
  /*
    * information about the document given
  */
  text: string[];
  classNames?: string[];
  localExplanations: number[];
  prediction?: number[];
}
