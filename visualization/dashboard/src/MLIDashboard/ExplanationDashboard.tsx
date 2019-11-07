import * as React from "react";
import { IExplanationDashboardProps } from "./Interfaces/IExplanationDashboardProps";
import { number } from "prop-types";
import { TextHighlighting } from "./Control/TextHightlighting";

export interface IDashboardContext {
}

export interface IDashboardState {
}

export class ExplanationDashboard extends React.Component<IExplanationDashboardProps, IDashboardState> {
    public static featureNamesValueImportanceDict(props:IExplanationDashboardProps):{[key: string]: number[]}[]{
        const dictList:{[key: string]: number[]}[]=[];
        let explanation = props.dataSummary.localExplanations;
        let featureNames = props.dataSummary.featureNames;
        // console.log(featureNames)
        // console.log(explanation)
        // console.log("---")
        featureNames.forEach((featureArray, featureArrayIndex) => {
            let dict:{[key: string]: number[]}={}
            featureArray.forEach((word, wordIdx) => {
                dict[word] = [];
                explanation.forEach((expArray) => {
                dict[word].push(expArray[featureArrayIndex][wordIdx]);
                });
            });
            dictList.push(dict);
        });
        return dictList;
    }

    public render() {
        let featureDict = ExplanationDashboard.featureNamesValueImportanceDict(this.props)
        return ( //look at how they do dataExploration
            <>
                    <h1>Interpretibility Dashboard</h1>
                    <div className="explainerDashboard">
                        Placeholder for the text explanation dashboard
                    </div>
                    {/* <p>{ExplanationDashboard.textHightlighting(this.props)}</p> */}
                    {/* {TextHighlighting.highlighting(dict, this.props)} */}
                    <TextHighlighting
                        dictList={featureDict}
                        data = {this.props}
                    />

                    
                </>
        );
    }
}
