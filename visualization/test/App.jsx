import React from 'react';
import {ExplanationDashboard}  from '../dashboard/rel/ExplanationDashboard';


    class App extends React.Component {
      constructor(props) {
        super(props);
        this.state = {value: 3};
      }

      render() {
        return (
          <div style={{backgroundColor: 'grey', height:'100%'}}>
              <div style={{ width: '80vw', backgroundColor: 'white', margin:'50px auto'}}>
                  <div style={{ width: '100%'}}>
                      <ExplanationDashboard
                      />
                  </div>
              </div>
          </div>
        );
      }
    }
    export default App;