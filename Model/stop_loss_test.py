######################### Imports #########################
import sys
from typing import Pattern
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.python.ops.gen_control_flow_ops import enter
######################### Feature Imports #########################
from feature_extraction import prepare_df, fit_pca, process_indicators, get_trend_change, add_alan_hull
######################### Attention Imports #########################
from attention import Attention
######################### StopLoss Imports #########################
from stop_loss_util import add_stop_price, successful_trades, oracle_test, add_price_level, prepare_trend_array, add_trend_columns, add_peak_trough_count
######################### Util Imports #########################
from research_util import root_mean_squared_error, custom_loss, mean_directional_accuracy


import os 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

plot_count = 0

def plot(actual_prices, inverse_pred_stop, inverse_pred_price, lookback_period, output_folder):
    global plot_count
    plt.figure(figsize=(19.20,10.80))
    plt.plot(actual_prices, color="green", label="Lookback")
    # current actual values
    off_s = lookback_period
    off_e = len(actual_prices)
    xaxis = [x for x in range(off_s, off_e)]
    yaxis = actual_prices[lookback_period:]
    plt.plot(xaxis, yaxis, color='red', label="Actual Closing Prices")
    # predicted stop values
    off_s = lookback_period
    off_e = len(actual_prices)
    xaxis = [x for x in range(off_s, off_e)]
    yaxis = inverse_pred_stop
    plt.plot(xaxis, yaxis, color='blue', label="Predicted Stop Prices")
    # predicted cp values
    off_s = lookback_period
    off_e = len(actual_prices)
    xaxis = [x for x in range(off_s, off_e)]
    yaxis = inverse_pred_price
    plt.plot(xaxis, yaxis, color='orange', label="Predicted Closing Prices")

    plt.title("Apple stock closing prices")
    plt.xlabel("Time")
    plt.ylabel("Stock price")
    plt.legend()
    plt.savefig(output_folder+"stop_res_fig" + str(plot_count) + ".png")

    plot_count += 1

def find_trend_trades(test_df, lookback_period, trade_window=10, number_of_trades=10, direction="up"):
    trade_count = 0
    trades_df = pd.DataFrame(columns=test_df.columns)
    i = 0
    while i < len(test_df.index) - (lookback_period+trade_window):
        if trade_count >= number_of_trades:
            break
        # lb_period defines the trend of the trade. last element is the value to be predicted
        end_index = i + lookback_period + trade_window
        trade_df = test_df.iloc[i:end_index]
        # Once the trade has started, it has to be of the direction given (previous data points in lookback doesn't matter)
        trend_arr = prepare_trend_array(trade_df[lookback_period-1:])
                
        trend_arr = [str(j) for j in trend_arr]
        trend_pattern = "".join(trend_arr)
        # if "43" not in trend_pattern and not trend_pattern.endswith("4") and not trend_pattern.endswith("3"):
        #     # plt.plot([i for i in range(lookback_period+1)], trade_df[["Close"]].values, color='orange', label="plt")
        #     # plt.savefig(output_folder+"plot.png")
        #     print(trend_pattern)
        # else:
        #     i += 1

        if direction == "up" and not "3" in trend_pattern[1:] and not "4" in trend_pattern[1:] and not trend_pattern.startswith("00"):
            # plt.plot([i for i in range(lookback_period+1)], trade_df[["Close"]].values, color='orange', label="plt")
            # plt.savefig(output_folder+"plot.png")
            print(trend_pattern)
            trades_df = trades_df.append(trade_df)
            i = end_index
            trade_count += 1
        elif direction == "down" and not "1" in trend_pattern[1:] and not "2" in trend_pattern[1:] and not trend_pattern.startswith("00"):
            print(trend_pattern)
            trades_df = trades_df.append(trade_df)
            i = end_index
            trade_count += 1
        else:
            i += 1

        # if "4" not in trend_pattern and "3" not in trend_pattern:
        #     # plt.plot([i for i in range(lookback_period+1)], trade_df[["Close"]].values, color='orange', label="plt")
        #     # plt.savefig(output_folder+"plot.png")
        #     print(trend_pattern)
        #     trades_df = trades_df.append(trade_df)
        #     i = end_index
        #     trade_count += 1 
        # else:
        #     i += 1
    
    return trades_df

def perform_directional_trades(test_df, model, scaler, lookback_period, trade_window=10, trend_index=-2, stop_price_index=-1, number_of_trades=10, direction="up"):
    actual_prices = []
    pred_prices = []
    pred_stops = []
    entering_prices = []

    trend_df = find_trend_trades(test_df, lookback_period, trade_window, number_of_trades, direction)
    trend_df = trend_df.reset_index(drop=True)
    # print(trend_df.to_string())
    print(len(trend_df.index))

    for i in range(0, number_of_trades):
        data_start_index = (i*trade_window) + (i*lookback_period)
        data_end_index = data_start_index + trade_window + lookback_period

        print(data_start_index, data_end_index)
        
        # print("start:end -", data_start_index, data_end_index)
        test_data = trend_df[data_start_index:data_end_index]
        
        pred_price, pred_stop = test(test_data, model, scaler, lookback_period, trend_index, stop_price_index)
        # print(test_data["Close"][lookback_period:].values.reshape(-1, trade_window))
        entering_price = trend_df["Close"][data_start_index + lookback_period-1]
        actual_price = test_data["Close"][lookback_period:].values.reshape(-1, trade_window)
        entering_prices.append(entering_price)
        actual_prices.append(actual_price)
        pred_prices.append(pred_price)
        pred_stops.append(pred_stop)
    
    entering_prices = np.array(entering_prices)
    actual_prices = np.array(actual_prices)
    pred_prices = np.array(pred_prices)
    pred_stops = np.array(pred_stops)

    actual_prices = actual_prices.reshape(-1, trade_window)
    pred_prices = pred_prices.reshape(-1, trade_window)
    pred_stops = pred_stops.reshape(-1, trade_window)
    entering_prices = entering_prices.reshape(-1, 1)

    # successful_trades(actual_prices, pred_prices, pred_stops, entering_prices, trade_window)
    oracle_test(actual_prices, pred_prices, pred_stops, entering_prices, trade_window)

def test(test_df, model, scaler, lookback_period, trend_index=-2, stop_price_index=-1):
    # get column indexes to seperate data for model inputs
    non_disc_col_indexes = test_df.columns.get_indexer([
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
    
    disc_col_indexes = test_df.columns.get_indexer([
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

    scaled_data = scaler.transform(test_df.values.reshape(-1, len(test_df.columns)))
    x_test = []
    x_test_price = []

    test_end_index = trend_index if trend_index < stop_price_index else stop_price_index

    for i in range(lookback_period, len(scaled_data)):
        x_test.append(scaled_data[i-lookback_period:i, :test_end_index])
        x_test_price.append(scaled_data[i-1, 0])

    x_test = np.array(x_test)
    x_test_price = np.array(x_test_price)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

    # split x_train to 2 seperate inputs
    # exlude non_disc columns for trend_detection
    x_test_tpred = np.delete(x_test, non_disc_col_indexes, axis=2)
    # exlude disc columns for price_prediction
    x_test_ppred = np.delete(x_test, disc_col_indexes, axis=2)

    x_test_price = x_test_price.reshape(-1, 1, 1)

    #actual prices
    actual_prices = test_df["Close"].values
    actual_stop_prices = test_df["stop_price"].values

    # predict
    pred_stop, pred_trend, pred_price = model.predict([x_test_ppred, x_test_tpred, x_test_price])

    dummy_data = np.zeros((pred_price.shape[0], len(test_df.columns) - 3))
    dummy_trend = np.zeros((pred_trend.shape[0], 1))

    dummy_scaler_data = np.append(pred_price, dummy_data, axis=1)
    dummy_scaler_data = np.append(dummy_scaler_data, dummy_trend, axis=1)
    dummy_scaler_data = np.append(dummy_scaler_data, pred_stop, axis=1)

    inverse_data = scaler.inverse_transform(dummy_scaler_data)

    inverse_pred_price = inverse_data[:, 0]
    inverse_pred_stop = inverse_data[:, stop_price_index]

    # Plot
    # plot(actual_prices, inverse_pred_stop, inverse_pred_price, lookback_period, output_folder)

    # print(inverse_pred_stop)
    # print(actual_stop_prices)

    # Calculate MDA of the result
    tf.print("MDA of the test is:", mean_directional_accuracy(actual_prices[lookback_period:], inverse_pred_stop.reshape(-1,1)))

    return inverse_pred_price, inverse_pred_stop

def perform_trades(test_df, model, scaler, lookback_period, trade_window=10, trend_index=-2, stop_price_index=-1, number_of_trades=10):
    actual_prices = []
    pred_prices = []
    pred_stops = []
    entering_prices = []

    for i in range(0, number_of_trades):
        data_start_index = i*trade_window
        data_end_index = (i+1)*trade_window + lookback_period
        # print("start:end -", data_start_index, data_end_index)
        test_data = test_df[data_start_index:data_end_index]
        
        pred_price, pred_stop = test(test_data, model, scaler, lookback_period, trend_index, stop_price_index)
        # print(test_data["Close"][lookback_period:].values.reshape(-1, trade_window))
        entering_price = test_df["Close"][data_start_index + lookback_period-1]
        actual_price = test_data["Close"][lookback_period:].values.reshape(-1, trade_window)
        entering_prices.append(entering_price)
        actual_prices.append(actual_price)
        pred_prices.append(pred_price)
        pred_stops.append(pred_stop)
    
    entering_prices = np.array(entering_prices)
    actual_prices = np.array(actual_prices)
    pred_prices = np.array(pred_prices)
    pred_stops = np.array(pred_stops)

    actual_prices = actual_prices.reshape(-1, trade_window)
    pred_prices = pred_prices.reshape(-1, trade_window)
    pred_stops = pred_stops.reshape(-1, trade_window)
    entering_prices = entering_prices.reshape(-1, 1)

    # successful_trades(actual_prices, pred_prices, pred_stops, entering_prices, trade_window)
    oracle_test(actual_prices, pred_prices, pred_stops, entering_prices, trade_window)

if __name__ == "__main__":
    arg_list = sys.argv

    if len(arg_list) < 7:
        sys.exit("Please input dataset,output_folder,model,scaler,pca_model,pca_scaler")
    
    lookback_period = 7
    trade_window = 10
    stop_alpha = 0.0001
    dataset_path = arg_list[1]
    output_folder = arg_list[2]
    model = load_model(arg_list[3], custom_objects={
                "Attention": Attention,
                "custom_loss": custom_loss,
                # "mean_directional_accuracy": mean_directional_accuracy
            })
    scaler = joblib.load(arg_list[4])
    # pc_model = joblib.load(arg_list[5])
    # pc_scaler = joblib.load(arg_list[6])

    if output_folder[-1] != "/":
        output_folder += "/"
    
    test_df = pd.read_csv(dataset_path)
    test_df = test_df[["Close", "Open", "High", "Low", "Volume"]]

    test_df = add_alan_hull(test_df, "Close", 10)
    test_df = prepare_df(test_df)
    # test_df = fit_pca(test_df, pc_model, pc_scaler, start_col_index=5)
    test_df = process_indicators(test_df,
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
    
    # # add price level (LL, LH, HL, HH)
    # test_df = add_price_level(test_df, "Close")
    test_df = add_trend_columns(test_df, "Close")

    # add peak and trough counts
    # test_df = add_peak_trough_count(test_df, "trend_hh", "trend_lh")

    # add y_trend
    test_df = get_trend_change(test_df, "Close")

    # add stop price
    test_df = add_stop_price(test_df, trade_window, stop_alpha)

    # print(test_df.head(10).to_string())

    # testing only for first 1004 (random)
    # test_df = test_df[:17]

    # pred_price, pred_stop = test(test_df, model, scaler, lookback_period)
    # plot(test_df["Close"].values, pred_stop, pred_price, lookback_period, output_folder)
    # print(test_df.head(17))
    # perform_trades(test_df, model, scaler, lookback_period, trade_window, -2, -1, 50)
    perform_directional_trades(test_df, model, scaler, lookback_period, trade_window, -2, -1, 10, "down")
    # print(len(test_df) // lookback_period+1)
    # trend_arr = find_uptrend_trades(test_df, lookback_period)
    # print(trend_arr)
