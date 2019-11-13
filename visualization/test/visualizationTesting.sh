#!/bin/bash
#testing visualizations from the vizualizations/test folder

cd ..
cd dashboard
yarn build
yarn build-css
cd ..
cd test
yarn build
yarn start