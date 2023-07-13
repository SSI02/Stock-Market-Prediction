import math
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers , regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Bidirectional
import matplotlib.dates as mdates
from tensorflow.keras.callbacks import EarlyStopping
import datetime as dt


def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')
    

def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    minimum = 4544.200195
    difference = 10770.500004999998
    
    columns = data.columns
    columns = columns[1:]
    # columns
    
    for column in columns:
        z = pd.Series(data[column])
        data[column] = z.interpolate(limit_direction='both', kind='cubic')
    
    X1 = np.array(data['Close'])
    X1 = X1.reshape((1,50))
    X1 = (X1-minimum)/difference
    
    X1 = X1.reshape((1, 50, 1))
    X1 = X1.astype(np.float32)
    X1 = np.array(X1)
    
    model = tf.keras.models.load_model('my_model.h5')

    pred_close = model.predict(X1).flatten()
    pred_close = (pred_close*difference) + minimum
    
    return pred_close
    
    

if __name__== "__main__":
    evaluate()