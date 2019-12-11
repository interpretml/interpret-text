// /* eslint-disable no-unused-expressions */
// import { Slider } from 'office-ui-fabric-react/lib/Slider'
// import { initializeIcons } from '@uifabric/icons';
// import { IStackTokens, Stack } from 'office-ui-fabric-react/lib/Stack'
// import React from 'react'
// import { number } from 'prop-types'
// import { IDatasetSummary } from '../Interfaces/IExplanationDashboardProps'

// // require("./SlidingBar.css")

// initializeIcons()

// export interface ISlidingBarState{
//     topK: number
// }
// class SlidingBar extends React.Component<{}, ISlidingBarState, IDatasetSummary> {
//     // constructor(props){
//     //     super(props);
//     // }
//     public state: ISlidingBarState = { topK: 0 };
//     public render(): {
//         const stackTokens: IStackTokens = { childrenGap: 20 };
//         return (
//             <Stack tokens={stackTokens} styles={{ root: { maxWidth: 300 } }}>
//               <Slider
//                 label="Basic example"
//                 min={1}
//                 max={5}
//                 step={1}
//                 defaultValue={2}
//                 showValue={true}
//                 onChange={(value: number) => console.log(value)}
//               />
//               </Stack>
//         );

//     }
// }
