import React from "react";
import { IExplanationDashboardProps } from "../Interfaces/IExplanationDashboardProps";

export interface ITextHighlighting {
    dictList :  {[key: string]: number[]}[]
    data: IExplanationDashboardProps
    }

export class TextHighlighting extends React.Component<ITextHighlighting>{
    public render(): React.ReactNode[]{ //control
        /* 
            function: goes through the passed in text and highlights the important features based on the importance values from dictionary
            input: props (data passed from Explanation Dashboard)
            output: list of html spam elements to render
        */
        let dict=this.props.dictList;
        let text=this.props.data.dataSummary.text;
        let label=this.props.data.dataSummary.predictionIndex;
        let textTokens= [];
        let returnText=[]; //how to declare this as a list of html spans?
        text.map((document, documentIndex)=>{
            textTokens.push(document.split(/ +/));
        })
        return textTokens.map((document, documentIndex)=>{
            const colorDocs = document.map((token, tokenIndex)=>{
                console.log(token.replace(/[^a-zA-Z0-9]+/g,""))
                let word = token.replace(/[^a-zA-Z0-9]+/g,"")
                if (word in dict[documentIndex]){
                    if (dict[documentIndex][word][documentIndex] > 0){
                        return <span className="makeitred" title={token}>{token+" "}</span>
                    }
                    else if (dict[documentIndex][word][documentIndex] < 0){
                        return <span className="makeitblue" title={token}>{token+" "}</span>
                    }
                    else{
                        return <span>{token+" "}</span>
                    }
                }
                else{
                    return <span>{token+" "}</span>
                }
    
            })
            return <div>{colorDocs}</div>
        }) 
    }
}

