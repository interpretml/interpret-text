import React from 'react';
import {ExplanationDashboard}  from '../dashboard/rel/ExplanationDashboard';

import {newsgroupBinaryData} from './_mock_data/newsgroupBinaryData';


    class App extends React.Component {
      constructor(props) {
        super(props);
      }

      render() {
        let data = newsgroupBinaryData;
        return (
          <div style={{backgroundColor: 'white', height:'100%'}}>
              <div style={{ width: '80vw', backgroundColor: 'white', margin:'50px auto'}}>
                  <div style={{ width: '100%'}}>
                      <ExplanationDashboard
                        modelInformation= {{modelInformation:"blackbox", method: "classifier"}}
                        //dataSummary = {{text: data.text, featureNames: data.featureNames, classNames: data.classNames, localExplanations: data.localExplanations, predictionIndex: data.predictionIndex}}
                        dataSummary = {{text: data.text, localExplanations: data.localExplanations, classNames: data.classNames, prediction:data.prediction}}
                      />
                  </div>
              </div>
          </div>
        );
      }
    }
    export default App;