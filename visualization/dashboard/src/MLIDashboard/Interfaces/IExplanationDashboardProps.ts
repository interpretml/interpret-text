export interface IExplanationDashboardProps {
    modelInformation: IModelInformation;
    dataSummary: IDatasetSummary;
    }

export interface IModelInformation {
    modelClass: 'blackbox';
    method: 'classifier' | 'regressor';
}

export interface IDatasetSummary {
    text: string[];
    classNames: string[];
    localExplanations: number[];
    prediction: number[];
}
