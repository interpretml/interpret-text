export interface IExplanationDashboardProps {
    modelInformation: IModelInformation;
    dataSummary: IDatasetSummary;
    config: IFeatureSelectionProps;
    onChange:(config:IFeatureSelectionProps)=>void;
    }

export interface IModelInformation {
    modelClass: 'msra'| 'bow' | 'rmp';
}

export interface IDatasetSummary {
    text: string[];
    classNames: string[];
    localExplanations: number[];
    prediction: number[];
    topK: number;
}

export interface IFeatureSelectionProps {
    topK: number;
}