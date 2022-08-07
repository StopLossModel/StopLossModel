######################### Imports #########################
import os 
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

######################### Feature Imports #########################
from feature_extraction import prepare_df, perform_pca, process_indicators, get_trend_change, add_alan_hull

######################### Util Imports #########################
from research_util import root_mean_squared_error, custom_loss, mean_directional_accuracy

######################### Attention Imports #########################
from attention import Attention

######################### StopLoss Imports #########################
from stop_loss import StopLossPrediction, StopLossPrediction2
from stop_loss_util import add_stop_price, add_price_level, add_trend_columns, add_peak_trough_count, calculate_alpha


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

######################### __main__ #########################
if __name__ == "__main__":
    arg_list = sys.argv
    if len(arg_list) < 3:
        sys.exit("Please input dataset and output_folder")

    lookback_period = 7
    epochs = 2000
    batch_size = 500
    trend_detection_window = 1
    trade_window = 10
    stop_alpha = 0.0001
    prediction_window = 5
    optimizer = "adam"
    model_name = "StopLoss2_LSTM_CNN_SBUX_10_1_1.0_1.0_cus_"

    print("Epochs:", epochs, " batch_size:", batch_size)
    
    dataset_path = arg_list[1]
    output_folder = arg_list[2]

    if output_folder[-1] != "/":
        output_folder += "/"

    dataframe = pd.read_csv(dataset_path)
    dataframe = dataframe[["Close", "Open", "High", "Low", "Volume"]]

    # dataframe = dataframe[-8000:]
    dataframe.dropna(inplace=True)

    print(len(dataframe.index))

    print("Train data length", len(dataframe.index))
    
    # Add TA, perform PCA and prepare dataframe

    dataframe = add_alan_hull(dataframe, "Close", 10)
    dataframe = prepare_df(dataframe)
    
    # sma, wma, stK, stD, will, macd, rsi, cci, ad, momentum
    dataframe = process_indicators(dataframe,
                "Close",
                "trend_sma_fast",
                "wma",
                "stoch_k",
                "stoch_d",
                "momentum_wr",
                "trend_macd",
                "momentum_rsi",
                "trend_cci",
                "volume_adi",
                "momentum"
                )

    # add price level (LL, LH, HL, HH)
    # dataframe = add_price_level(dataframe, "Close")
    dataframe = add_trend_columns(dataframe, "Close")

    # add peak and trough counts
    # dataframe = add_peak_trough_count(dataframe, "trend_hh", "trend_lh")

    # add y_trend
    dataframe = get_trend_change(dataframe, "Close")

    # calculate diff alpha
    stop_alpha = calculate_alpha(dataframe["Close"])

    # add stop price
    dataframe = add_stop_price(dataframe, trade_window, stop_alpha)

    # get column indexes to seperate data for model inputs
    non_disc_col_indexes = dataframe.columns.get_indexer([
                                "trend_sma_fast",
                                "wma",
                                "stoch_k",
                                "stoch_d",
                                "momentum_wr",
                                "trend_macd",
                                "momentum_rsi",
                                "trend_cci",
                                "volume_adi",
                                "momentum"
                            ])
    
    disc_col_indexes = dataframe.columns.get_indexer([
                                "pr_sma",
                                "pr_wma",
                                "pr_stK",
                                "pr_stD",
                                "pr_will",
                                "pr_macd",
                                "pr_rsi",
                                "pr_cci",
                                "pr_ad",
                                "pr_moment"
                            ])

    # Scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # increase the range of min and max values by 25% to cover possible test data ranges
    scaling_values = dataframe.values.reshape(-1, len(dataframe.columns))
    max_values = scaling_values.max(axis=0, keepdims=True)[0]
    min_values = scaling_values.min(axis=0, keepdims=True)[0]

    max_indexes = scaling_values.argmax(axis=0)
    min_indexes = scaling_values.argmin(axis=0)

    # update values by 25%
    for i in range(0, len(dataframe.columns)):
        scaling_values[max_indexes[i]][i] = max_values[i] + max_values[i] * 0.25
        scaling_values[min_indexes[i]][i] = min_values[i] - min_values[i] * 0.25

    # fit scaler
    scaler.fit(scaling_values)

    # transform training values
    scaled_data = scaler.transform(dataframe.values.reshape(-1, len(dataframe.columns)))

    # separate x_data, y_data
    x_train = []
    x_train_price = []
    y_train_price = []      # col index = 0
    y_train_trend = []      # col index = -2
    y_train_stop = []       # col index = -1

    for i in range(lookback_period, len(scaled_data)):
        x_train.append(scaled_data[i-lookback_period:i, :-2])
        x_train_price.append(scaled_data[i-1, 0])
        y_train_price.append(scaled_data[i, 0])
        y_train_stop.append(scaled_data[i, -1])
    
    # y_trend
    for i in range(lookback_period+trend_detection_window, len(scaled_data) + 1):
        y_train_trend.append(scaled_data[i-trend_detection_window:i, -2])

    if trend_detection_window > 1:
        # dropping last trend_detection_window - 1 amount of tail data to match trend data count (change later)
        x_train = x_train[:-(trend_detection_window-1)]
        x_train_price = x_train_price[:-(trend_detection_window-1)]
        y_train_price = y_train_price[:-(trend_detection_window-1)]
        y_train_stop = y_train_stop[:-(trend_detection_window-1)]
        ################

    x_train, x_train_price, y_train_price = np.asarray(x_train), np.asarray(x_train_price), np.asarray(y_train_price)
    y_train_trend, y_train_stop = np.asarray(y_train_trend), np.asarray(y_train_stop)

    print(x_train.shape)

    # split x_train to 2 seperate inputs
    # exlude non_disc columns for trend_detection
    x_train_tpred = np.delete(x_train, non_disc_col_indexes, axis=2)

    # exlude disc columns for price_prediction
    x_train_ppred = np.delete(x_train, disc_col_indexes, axis=2)

    x_train_tpred = x_train_tpred.reshape(x_train_tpred.shape[0], x_train_tpred.shape[1], x_train_tpred.shape[2])
    x_train_ppred = x_train_ppred.reshape(x_train_ppred.shape[0], x_train_ppred.shape[1], x_train_ppred.shape[2])
    x_train_price = x_train_price.reshape(-1, 1, 1)
    y_train_price = y_train_price.reshape(-1, 1, 1)

    print(x_train_tpred.shape)
    print(x_train_ppred.shape)
    print(x_train_price.shape)
    print(y_train_price.shape)
    print(y_train_stop.shape)
    print(y_train_trend.shape)

    x_shape_trend = (x_train_tpred.shape[1], x_train_tpred.shape[2])
    x_shape_price = (x_train_ppred.shape[1], x_train_ppred.shape[2])
    price_shape = (x_train_price.shape[1], x_train_price.shape[2])

    enc_model, dec_model = StopLossPrediction2.build(x_shape_price, x_shape_trend, price_shape, trend_detection_window)

    print(dec_model.summary())

    # training model
    dec_model.fit([x_train_ppred, x_train_tpred, x_train_price], [y_train_stop, y_train_trend, y_train_price.reshape(-1,1)], epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=2)

    # save model and scaler
    model_path_name = output_folder+model_name+str(stop_alpha)+".h5"
    scaler_path_name = output_folder+model_name+str(stop_alpha)+"_scaler.scaler"
    print("Saving model to", model_path_name)
    dec_model.save(model_path_name)
    print("Saving scaler to", scaler_path_name)
    joblib.dump(scaler, scaler_path_name)
    print("model_name:", model_name)

