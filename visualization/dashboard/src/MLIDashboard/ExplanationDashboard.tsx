import * as React from "react";

export interface IDashboardContext {
}

export interface IDashboardState {
}

export class ExplanationDashboard extends React.Component<IDashboardState> {
    public render() {
        return (
            <>
                <div className="explainerDashboard">
                    Placeholder for the text explanation dashboard
                </div>
            </>
        );
    }
}