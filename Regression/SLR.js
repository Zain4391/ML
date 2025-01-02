const fs = require('fs');
const csv = require('csv-parser');
const LinearRegression = require('ml-regression').SimpleLinearRegression;
const { plot, Plot } = require('nodeplotlib');

// We construct the features matrix and dependent variable vector
let X = [];
let y = [];

fs.createReadStream('house_prices.csv')
.pipe(csv())
.on('data', (row) => {
    X.push(parseFloat(row['House Size (sq ft)']));
    y.push(parseFloat(row['House Price (in $1000s)']));
})
.on('end', () => {
    console.log('CSV file read');
    
    // Separate the data into training and test sets
    const splitIdx = Math.floor(X.length * 0.8);
    const X_train = X.slice(0, splitIdx);
    const y_train = y.slice(0, splitIdx);
    const X_test = X.slice(splitIdx);
    const y_test = y.slice(splitIdx);

    // Instantiate the model and train it
    const regressor = new LinearRegression(X_train, y_train);
    const slope = regressor.slope;
    const intercept = regressor.intercept;

    const equation = `Prices = ${slope}X + ${intercept}`;
    console.log(equation);

    // Predict values
    let y_pred = [];
    X_test.forEach((value) => {
        y_pred.push(regressor.predict(value));
    });

    console.log("\nPREDICTIONS:");
    y_pred.forEach((value, index) => {
        console.log(`House Size: ${X_test[index]} predicted value: ${value.toFixed(2)}, Actual value: ${y_test[index]}`);
    });

    // Calculate evaluation metric (MSE)
    let mse = 0.0;
    let ss_total = 0.0;    // Total sum of squares
    let ss_residual = 0.0; // Residual sum of squares
    const mean_y_test = y_test.reduce((sum, val) => sum + val, 0) / y_test.length; // Mean of y_test

    y_pred.forEach((value, index) => {
        let residual = y_test[index] - value;
        residual = Math.pow(residual, 2);
        mse += residual;

        // for R^2
        ss_residual += Math.pow(y_test[index] - value, 2);
        ss_total += Math.pow(y_test[index] - mean_y_test, 2);
    });

    mse = mse / y_test.length;
    const r2 = 1 - (ss_residual/ ss_total);
    console.log(`Computed MSE: ${mse.toFixed(2)}`);
    console.log(`Computed R²: ${r2.toFixed(4)}`); // Print R²

    // --------------------------------
    // PLOT 1: Training Data Plot
    // --------------------------------
    const trainingData = [
        {
            x: X_train,
            y: y_train,
            mode: 'markers',
            name: 'Training Data',
            marker: { color: 'red' }
        },
        {
            x: X_train,
            y: regressor.predict(X_train),
            mode: 'lines',
            name: 'Regression Line',
            line: { color: 'blue' }
        }
    ];

    const trainingLayout = {
        title: 'Training Data - House Prices vs House Sizes',
        xaxis: { title: 'House Size (sq ft)' },
        yaxis: { title: 'House Price (in $1000s)' }
    };

    // Display Training Data Plot
    plot(trainingData, trainingLayout);

    // --------------------------------
    // PLOT 2: Test Data Plot
    // --------------------------------
    const testData = [
        {
            x: X_test,
            y: y_test,
            mode: 'markers',
            name: 'Test Data',
            marker: { color: 'green' }
        },
        {
            x: X_test,
            y: y_pred,
            mode: 'lines',
            name: 'Predictions',
            line: { color: 'blue' }
        }
    ];

    const testLayout = {
        title: 'Test Data - Predictions vs Actual Values',
        xaxis: { title: 'House Size (sq ft)' },
        yaxis: { title: 'House Price (in $1000s)' }
    };

    // Display Test Data Plot
    plot(testData, testLayout);
});
