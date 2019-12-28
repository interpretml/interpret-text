export interface IExplanationDashboardProps {
    modelInformation: IModelInformation;
    dataSummary: IDatasetSummary;
    }

export interface IModelInformation {
    modelClass: 'msra'| 'bow' | 'rmp';
}

export interface IDatasetSummary {
    text: string[];
    classNames?: string[];
    localExplanations: number[];
    prediction?: number[];
    topK?: number;
}
