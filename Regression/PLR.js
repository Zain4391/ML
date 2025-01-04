// Import required libraries
const fs = require('fs');
const csv = require('csv-parser');
const ml = require('ml-regression');
const PolynomialRegression = ml.PolynomialRegression;
const { plot } = require('nodeplotlib');

// Load and parse the dataset
const X = [];
const y = [];

fs.createReadStream('vehicles.csv')
  .pipe(csv())
  .on('data', (row) => {
    X.push(parseFloat(row['Battery Capacity (kWh)'])); // Battery Capacity
    y.push(parseFloat(row['Price ($1000)'])); // Price
  })
  .on('end', () => {
    // Split the dataset into training and testing sets (80-20 split)
    const splitIndex = Math.floor(0.8 * X.length);
    const X_train = X.slice(0, splitIndex);
    const y_train = y.slice(0, splitIndex);
    const X_test = X.slice(splitIndex);
    const y_test = y.slice(splitIndex);

    // Train the Polynomial Regression Model
    const degree = 4;
    const polyReg = new PolynomialRegression(X_train, y_train, degree);

    // Predictions
    const y_pred = X_test.map(x => polyReg.predict(x));

    // Evaluate the Model
    const mse = y_test.reduce((sum, yi, i) => sum + Math.pow(yi - y_pred[i], 2), 0) / y_test.length;
    const r2 = 1 - (mse / y_test.reduce((sum, yi) => sum + Math.pow(yi - (y_test.reduce((a, b) => a + b) / y_test.length), 2), 0));

    console.log(`Mean Squared Error (MSE): ${mse.toFixed(2)}`);
    console.log(`R-squared (R²): ${r2.toFixed(4)}`);

    // Visualization
    const sortedData = X.map((x, i) => ({ x, y: polyReg.predict(x) })).sort((a, b) => a.x - b.x);
    plot([
        { x: X, y: y, type: 'scatter', mode: 'markers', name: 'Actual Data', marker: { color: 'red' } },
        { x: sortedData.map(d => d.x), y: sortedData.map(d => d.y), type: 'scatter', mode: 'lines', name: 'Polynomial Fit', line: { color: 'blue' } }
    ], {
        title: 'Vehicle Price Prediction',
        xaxis: { title: 'Battery Capacity (kWh)' },
        yaxis: { title: 'Price ($1000)' }
    });
  });
