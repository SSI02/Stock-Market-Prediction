# Stock Price Prediction

This project predicts the closing prices for a stock index using an LSTM model. It provides a sample evaluation function and includes instructions for evaluating the model's performance.

## Instructions

1. Clone the repository or download the code files.

2. Install the required libraries by running the following command:

3. Prepare the data:
- Ensure you have the following files in the same directory as the code:
  - `my_model.h5`: Trained LSTM model file.
  - `sample_input.csv`: CSV file containing the time series data for the stock index.
  - `sample_close.txt`: Text file containing the actual closing prices for evaluation.

4. Run the evaluation script by executing the following command:
The script will load the `sample_input.csv` file, make predictions using the LSTM model, and evaluate the predictions against the actual closing prices provided in `sample_close.txt`.

5. The evaluation results will be displayed, showing the mean square error and directional accuracy of the predictions.

6. `Jupyter_File.ipynb` is the jupyter notebook in which the model architecture, hyperparameters, and data preprocessing is done and all the model architecture is saved in file `my_model.h5`. Use this file for checking my Coding Efforts.

## Libraries Used

- pandas
- matplotlib
- datetime
- numpy
- scikit-learn
- tensorflow