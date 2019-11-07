import { logicalExpression } from "@babel/types";

export interface IExplanationDashboardProps {
    modelInformation: IModelInformation;
    dataSummary: IDatasetSummary;
    }

export interface IModelInformation {
    modelClass: "blackbox";
    method: "classifier" | "regressor";
}

export interface IDatasetSummary {
    text: string[];
    featureNames: string[];
    classNames: string[];
    localExplanations: any[][][];
    predictionIndex: number;
}
