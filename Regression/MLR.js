const fs = require('fs');
const csv = require('csv-parser');
const { MultivariateLinearRegression } = require('ml-regression');

// Data arrays
let X = [];
let y = [];

fs.createReadStream('student_scores_dataset.csv')
  .pipe(csv())
  .on('data', (row) => {
    if (
      row['Study_Hours'] &&
      row['Previous_Scores'] &&
      row['Attendance'] &&
      row['Final_Exam_Score']
    ) {
      X.push([
        parseFloat(row['Study_Hours']),
        parseFloat(row['Previous_Scores']),
        parseFloat(row['Attendance'])
      ]);
      y.push([parseFloat(row['Final_Exam_Score'])]); // Make y 2D array
    }
  })
  .on('end', () => {
    console.log('CSV file read');

    if (X.length === 0 || y.length === 0) {
      console.error('Error: No valid data found.');
      return;
    }

    // Split data into training and test sets
    const splitIdx = Math.floor(X.length * 0.8);
    const X_train = X.slice(0, splitIdx);
    const y_train = y.slice(0, splitIdx); // Already 2D
    const X_test = X.slice(splitIdx);
    const y_test = y.slice(splitIdx).flat(); // Flatten y_test for evaluation

    // Validate dimensions
    if (X_train.length === 0 || X_train[0].length === 0 || y_train.length === 0) {
      console.error('Error: Invalid data dimensions.');
      return;
    }

    // Train the model
    const regressor = new MultivariateLinearRegression(X_train, y_train);

    console.log(`Coefficients: ${regressor.weights}`);
    let y_pred = [];
    X_test.forEach((value) => {
        y_pred.push(regressor.predict(value));
    });

    console.log("\nPREDICTIONS:");
    y_pred.forEach((value, index) => {
      console.log(`Test Data: ${X_test[index]} predicted: ${value}, Actual: ${y_test[index]}`);
    });

    // Evaluate the model
    let mse = 0.0;
    let ss_total = 0.0;
    let ss_residual = 0.0;
    const mean_y_test = y_test.reduce((sum, val) => sum + val, 0) / y_test.length;

    y_pred.forEach((value, index) => {
      let residual = y_test[index] - value;
      mse += Math.pow(residual, 2);

      ss_residual += Math.pow(y_test[index] - value, 2);
      ss_total += Math.pow(y_test[index] - mean_y_test, 2);
    });

    mse /= y_test.length;
    const r2 = 1 - (ss_residual / ss_total);

    console.log(`Computed MSE: ${mse.toFixed(2)}`);
    console.log(`Computed R²: ${r2.toFixed(4)}`);
  });
