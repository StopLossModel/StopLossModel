from os import close
import pandas as pd
import numpy as np

######################### Util Functions #######################
def prepare_trend_array(df):
    trend_arr = []
    close_vals = df[["Close"]].values
    if len(close_vals) == 0:
        return trend_arr

    pre_val = close_vals[0]
    hh_val = pre_val
    hl_val = pre_val

    # HH, HL= 1,2 / LH, LL = 3,4 / Neut = 0

    trend_arr.append(0)
    for i in range(1, len(close_vals)):
        cur_val = close_vals[i]
        # high
        if cur_val > pre_val:
            # HH
            if cur_val >= hh_val:
                trend_arr.append(1)
                hh_val = cur_val
            # LH
            else:
                trend_arr.append(3)

        # low
        elif cur_val < pre_val:
            # HL
            if cur_val >= hl_val:
                trend_arr.append(2)
                hl_val = cur_val
            # LL
            else:
                trend_arr.append(4)

        # neut
        else:
            trend_arr.append(0)
        
        pre_val = cur_val

    if len(trend_arr) > 0:
        val = trend_arr[1]
        if val == 1 or val == 3:
            # label first value as HH
            trend_arr[0] = 1
        elif val == 2 or val == 4:
            # label first value as HL
            trend_arr[0] = 2
    
    return trend_arr
        
def calculate_alpha(close_df):
    mean_dif = close_df.diff().abs().sum() / (len(close_df.index) - 1)
    mean_close = close_df.mean()

    alpha = mean_dif / mean_close
    print("alpha:", alpha)
    return alpha

def add_stop_price(df, trade_window=10, alpha=0.1):
    print("Adding stop price")
    # alpha = calculate_alpha(df["Close"])
    close_vals = df[["Close"]].values
    
    # print(close_vals.shape)

    def find_max_index(values):
        index = values.argmax(axis=0)[0]
        return index

    exit_points = np.array([], dtype=int)

    for i in range(0, len(close_vals) // trade_window):
        window_values = close_vals[i*trade_window:(i+1)*trade_window]
        # pattern = [0] * len(window_values)

        max_index = find_max_index(window_values)

        max_value = window_values[max_index]
        # increase the value of the exit point
        window_values[max_index] = max_value + max_value * alpha
        
        # decrease the values of other entries
        for i in range(len(window_values)):
            if i != max_index:
                val = window_values[i]
                window_values[i] = val - val * alpha

        exit_points = np.concatenate((exit_points, window_values), axis=None)

    # add the values which are in the end that exceeds the trade_window
    if (len(close_vals) % trade_window) != 0:
        window_values = close_vals[-(len(close_vals) % trade_window):]

        # decrease the values of other entries
        for i in range(len(window_values)):
            val = window_values[i]
            window_values[i] = val - val * alpha
        
        exit_points = np.concatenate((exit_points, window_values), axis=None)

    df["stop_price"] = exit_points

    return df

def add_price_level(df: pd.DataFrame, close: str):
    close_vals = df[["Close"]].values
    trends = []
    hh = -1
    hl = -1
    prev_cp = -1
    # LL - 1, HL - 2, LH - 3, HH - 4
    for cp in close_vals:
        if cp > prev_cp: # high
            if cp > hh:
                trends.append(4)
                hh = cp
            else:
                trends.append(3)
        else:   # low
            if cp > hl:
                trends.append(2)
                hl = cp
            else:
                trends.append(1)

        prev_cp = cp
    
    df["price_level"] = np.array(trends)

    return df

def add_trend_columns(df: pd.DataFrame, close: str):
    hh_arr = []
    hl_arr = []
    lh_arr = []
    trend_arr = prepare_trend_array(df)

    index = 0
    for i in trend_arr:
        if i == 1:
            hh_arr.append(1)
            hl_arr.append(0)
            lh_arr.append(0)
        elif i==2:
            hh_arr.append(0)
            hl_arr.append(1)
            lh_arr.append(0)
        elif i == 3:
            hh_arr.append(0)
            hl_arr.append(0)
            lh_arr.append(1)
        elif i == 4:
            hh_arr.append(0)
            hl_arr.append(0)
            lh_arr.append(0)
        else:
            if index != 0:
                hh_arr.append(hh_arr[index-1])
                hl_arr.append(hl_arr[index-1])
                lh_arr.append(lh_arr[index-1])
            else:
                hh_arr.append(0)
                hl_arr.append(0)
                lh_arr.append(0)
        
        index += 1

    # label the first element(s) -> HL or HH
    if len(trend_arr) > 1:
        val = trend_arr[1]
        if val == 1 or val == 3:
            # label first value as HH
            hh_arr[0] = 1
        elif val == 2 or val == 4:
            # label first value as HL
            hl_arr[0] = 1

    df["trend_hh"] = np.array(hh_arr)
    df["trend_hl"] = np.array(hl_arr)
    df["trend_lh"] = np.array(lh_arr)  

    return df    

def add_peak_trough_count(df: pd.DataFrame, trend_hh: str, trend_lh: str):
    peak_count = 0
    trough_count = 0
    peaks = []
    troughs = []
    
    for i in range(0, len(df.index)):
        # add past peak/trough count
        peaks.append(peak_count)
        troughs.append(trough_count)

        # calculate new counts
        row = df.iloc[i]
        hh_val = row[trend_hh]
        lh_val = row[trend_lh]

        if hh_val == 1 or lh_val == 1:
            peak_count += 1
            trough_count = 0    # reset count on reversal
        else:
            trough_count += 1
            peak_count = 0      # reset count on reversal
    
    df["prev_peaks"] = peaks
    df["prev_troughs"] = troughs

    return df



def trailing_stop_profit(actual_prices, entering_price, percent):
    stop_price = entering_price - entering_price * percent
    prev_price = entering_price

    for price in actual_prices:
        if price <= stop_price:
            return price - entering_price
        if price > prev_price:
            stop_price = price - price * percent
    
    return actual_prices[-1] - entering_price    # return profit from last step of the trade if stop price didn't cross 

def fixed_stop_profit(actual_prices, entering_price, percent):
    stop_price = entering_price - entering_price * percent

    for price in actual_prices:
        if price <= stop_price:
            return price - entering_price
    
    return actual_prices[-1] - entering_price    # return profit from last step of the trade if stop price didn't cross

def std_dev_profit(actual_prices, entering_price):
    avg_price = actual_prices.mean()
    alpha = actual_prices.std()
    stop_price = avg_price - alpha

    for price in actual_prices:
        if price <= stop_price:
            return price - entering_price
    
    return actual_prices[-1] - entering_price    # return profit from last step of the trade if stop price didn't cross

# shapes = -1,trade_window
# entering_prices_shape = -1, 1
def successful_trades(actual_prices, pred_prices, pred_stops, entering_prices, trade_window=10):
    success_count = 0
    stop_exit_count = 0
    profits_made = []
    max_profits = []
    buy_and_hold_profits = []
    std_dev_profits = []
    zero_one_percent_fixed_stop_profits = []
    zero_two_percent_fixed_stop_porfits = []
    zero_five_percent_fixed_stop_profits = []
    one_percent_fixed_stop_profits = []
    zero_one_percent_trailing_stop_profits = []
    zero_two_percent_trailing_stop_porfits = []
    zero_five_percent_trailing_stop_profits = []
    one_percent_trailing_stop_profits = []
    for i in range(len(actual_prices)):
        actual_price = actual_prices[i, :]
        pred_price = pred_prices[i,:]
        pred_stop = pred_stops[i, :]
        entering_price = entering_prices[i, 0]

        max_price_of_trade_index = actual_price.argmax(axis=0)
        max_price_of_trade = actual_price[max_price_of_trade_index]

        max_profit_of_trade = max_price_of_trade - entering_price # if negative, then this is the least loss
        max_profits.append(max_profit_of_trade)

        buy_and_hold_profits.append(actual_price[-1] - entering_price)

        # standard deviation method profits
        std_dev_profits.append(std_dev_profit(actual_price, entering_price))

        # Fixed stop-loss profits
        zero_one_percent_fixed_stop_profits.append(fixed_stop_profit(actual_price, entering_price, 0.001))
        zero_two_percent_fixed_stop_porfits.append(fixed_stop_profit(actual_price, entering_price, 0.002))
        zero_five_percent_fixed_stop_profits.append(fixed_stop_profit(actual_price, entering_price, 0.005))
        one_percent_fixed_stop_profits.append(fixed_stop_profit(actual_price, entering_price, 0.01))

        # Trailing stop-loss profits
        zero_one_percent_trailing_stop_profits.append(trailing_stop_profit(actual_price, entering_price, 0.001))
        zero_two_percent_trailing_stop_porfits.append(trailing_stop_profit(actual_price, entering_price, 0.002))
        zero_five_percent_trailing_stop_profits.append(trailing_stop_profit(actual_price, entering_price, 0.005))
        one_percent_trailing_stop_profits.append(trailing_stop_profit(actual_price, entering_price, 0.01))

        # iterate the predictions and find exit point
        exit_point = -1
        for j in range(len(pred_price)):
            price = pred_price[j]
            stop = pred_stop[j]
            if stop >= price:
                exit_point = j
                break
        
        if exit_point != -1:
            if exit_point == max_price_of_trade_index:
                success_count += 1
                profits_made.append(max_profit_of_trade)
            else:
                profits_made.append(actual_price[exit_point] - entering_price)
            stop_exit_count += 1
        else:
            # didn't exit trade -> exits on last time step of trading window
            profits_made.append(actual_price[-1] - entering_price)

    print("Total number of trades:", len(actual_prices))
    print("Successful trades with max profit:", success_count)
    print("Trades exit by crossing stop price:", stop_exit_count)
    print("Accuracy:", success_count/len(actual_prices) * 100)

    total_investment = np.sum(entering_prices)

    max_profit = sum(max_profits)
    actual_profit = sum(profits_made)
    bnh_profit = sum(buy_and_hold_profits)
    stdDev_profit = sum(std_dev_profits)
    z_onep_fix_profit = sum(zero_one_percent_fixed_stop_profits)
    z_twop_fix_profit = sum(zero_two_percent_fixed_stop_porfits)
    z_fivep_fix_profit = sum(zero_five_percent_fixed_stop_profits)
    onep_fix_profit = sum(one_percent_fixed_stop_profits)
    z_onep_trail_profit = sum(zero_one_percent_trailing_stop_profits)
    z_twop_trail_profit =  sum(zero_two_percent_trailing_stop_porfits)
    z_fivep_trail_profit = sum(zero_five_percent_trailing_stop_profits)
    onep_trail_profit = sum(one_percent_trailing_stop_profits)

    print("Total investment:", total_investment)

    print("Maximum profit to be made from trades:", max_profit)
    print("Actual profit made from trades:", actual_profit)
    print("Buy and Hold profit:", bnh_profit)
    print("Standard Deviation Method profit:", stdDev_profit)
    print("0.1% fixed stop_loss profit:", z_onep_fix_profit)
    print("0.2% fixed stop_loss profit:", z_twop_fix_profit)
    print("0.5% fixed stop_loss profit:", z_fivep_fix_profit)
    print("1% fixed stop_loss profit:", onep_fix_profit)
    print("0.1% trailing stop_loss profit:", z_onep_trail_profit)
    print("0.2% trailing stop_loss profit:", z_twop_trail_profit)
    print("0.5% trailing stop_loss profit:", z_fivep_trail_profit)
    print("1% trailing stop_loss profit:", onep_trail_profit)
    
    print("\nPerformance against max profit\n")

    print("Actual profit/Max profit:", actual_profit/max_profit)
    print("Buy and Hold profit/Max profit:", bnh_profit/max_profit)
    print("Standard deviation profit/Max profit:", stdDev_profit/max_profit)
    print("0.1% fixed stop_loss profit/Max profit:", z_onep_fix_profit/max_profit)
    print("0.2% fixed stop_loss profit/Max profit:", z_twop_fix_profit/max_profit)
    print("0.5% fixed stop_loss profit/Max profit:", z_fivep_fix_profit/max_profit)
    print("1% fixed stop_loss profit/Max profit:", onep_fix_profit/max_profit)
    print("0.1% trailing stop_loss profit/Max profit:", z_onep_trail_profit/max_profit)
    print("0.2% trailing stop_loss profit/Max profit:", z_twop_trail_profit/max_profit)
    print("0.5% trailing stop_loss profit/Max profit:", z_fivep_trail_profit/max_profit)
    print("1% trailing stop_loss profit/Max profit:", onep_trail_profit/max_profit)

    print("\nProfit against investment\n")
    print("Maximum profit/investment:", max_profit/total_investment)
    print("Actual profit/investment:", actual_profit/total_investment)
    print("Buy and Hold profit/investment:", bnh_profit/total_investment)
    print("Standard deviatio profit/investment:", stdDev_profit/total_investment)
    print("0.1% fixed stop_loss profit/investment:", z_onep_fix_profit/total_investment)
    print("0.2% fixed stop_loss profit/investment:", z_twop_fix_profit/total_investment)
    print("0.5% fixed stop_loss profit/investment:", z_fivep_fix_profit/total_investment)
    print("1% fixed stop_loss profit/investment:", onep_fix_profit/total_investment)
    print("0.1% trailing stop_loss profit/investment:", z_onep_trail_profit/total_investment)
    print("0.2% trailing stop_loss profit/investment:", z_twop_trail_profit/total_investment)
    print("0.5% trailing stop_loss profit/investment:", z_fivep_trail_profit/total_investment)
    print("1% trailing stop_loss profit/investment:", onep_trail_profit/total_investment)
    # print(max_profits)
    # print(profits_made)
    # print(buy_and_hold_profits)
    # print(zero_one_percent_fixed_stop_profits)
    # print(zero_two_percent_fixed_stop_porfits)
    # print(zero_one_percent_trailing_stop_profits)
    # print(zero_two_percent_trailing_stop_porfits)

def oracle_test(actual_prices, pred_prices, pred_stops, entering_prices, trade_window=10):
    success_count = 0
    stop_exit_count = 0
    profits_made = []
    max_profits = []
    buy_and_hold_profits = []
    std_dev_profits = []
    zero_one_percent_fixed_stop_profits = []
    zero_two_percent_fixed_stop_porfits = []
    zero_five_percent_fixed_stop_profits = []
    one_percent_fixed_stop_profits = []
    zero_one_percent_trailing_stop_profits = []
    zero_two_percent_trailing_stop_porfits = []
    zero_five_percent_trailing_stop_profits = []
    one_percent_trailing_stop_profits = []
    for i in range(len(actual_prices)):
        actual_price = actual_prices[i, :]
        pred_price = pred_prices[i,:]
        pred_stop = pred_stops[i, :]
        entering_price = entering_prices[i, 0]

        max_price_of_trade_index = actual_price.argmax(axis=0)
        max_price_of_trade = actual_price[max_price_of_trade_index]

        max_profit_of_trade = max_price_of_trade - entering_price # if negative, then this is the least loss
        max_profits.append(max_profit_of_trade)

        buy_and_hold_profits.append(actual_price[-1] - entering_price)

        # standard deviation method profits
        std_dev_profits.append(std_dev_profit(actual_price, entering_price))

        # Fixed stop-loss profits
        zero_one_percent_fixed_stop_profits.append(fixed_stop_profit(actual_price, entering_price, 0.001))
        zero_two_percent_fixed_stop_porfits.append(fixed_stop_profit(actual_price, entering_price, 0.002))
        zero_five_percent_fixed_stop_profits.append(fixed_stop_profit(actual_price, entering_price, 0.005))
        one_percent_fixed_stop_profits.append(fixed_stop_profit(actual_price, entering_price, 0.01))

        # Trailing stop-loss profits
        zero_one_percent_trailing_stop_profits.append(trailing_stop_profit(actual_price, entering_price, 0.001))
        zero_two_percent_trailing_stop_porfits.append(trailing_stop_profit(actual_price, entering_price, 0.002))
        zero_five_percent_trailing_stop_profits.append(trailing_stop_profit(actual_price, entering_price, 0.005))
        one_percent_trailing_stop_profits.append(trailing_stop_profit(actual_price, entering_price, 0.01))

        # iterate the predictions and find exit point
        exit_point = -1
        for j in range(len(pred_price)):
            price = pred_price[j]
            stop = pred_stop[j]
            if stop >= price:
                exit_point = j
                break
        
        if exit_point != -1:
            if exit_point < max_price_of_trade_index:
                profits_made.append(actual_price[exit_point] - entering_price)
            else:
                profits_made.append(max_profit_of_trade)
        else:
            profits_made.append(max_profit_of_trade)

    print("Total number of trades:", len(actual_prices))

    total_investment = np.sum(entering_prices)
    max_profit = sum(max_profits)
    actual_profit = sum(profits_made)
    bnh_profit = sum(buy_and_hold_profits)
    stdDev_profit = sum(std_dev_profits)
    z_onep_fix_profit = sum(zero_one_percent_fixed_stop_profits)
    z_twop_fix_profit = sum(zero_two_percent_fixed_stop_porfits)
    z_fivep_fix_profit = sum(zero_five_percent_fixed_stop_profits)
    onep_fix_profit = sum(one_percent_fixed_stop_profits)
    z_onep_trail_profit = sum(zero_one_percent_trailing_stop_profits)
    z_twop_trail_profit =  sum(zero_two_percent_trailing_stop_porfits)
    z_fivep_trail_profit = sum(zero_five_percent_trailing_stop_profits)
    onep_trail_profit = sum(one_percent_trailing_stop_profits)

    print("Total investment:", total_investment)

    print("Maximum profit to be made from trades:", max_profit)
    print("Actual profit made from trades:", actual_profit)
    print("Buy and Hold profit:", bnh_profit)
    print("Standard Deviation Method profit:", stdDev_profit)
    print("0.1% fixed stop_loss profit:", z_onep_fix_profit)
    print("0.2% fixed stop_loss profit:", z_twop_fix_profit)
    print("0.5% fixed stop_loss profit:", z_fivep_fix_profit)
    print("1% fixed stop_loss profit:", onep_fix_profit)
    print("0.1% trailing stop_loss profit:", z_onep_trail_profit)
    print("0.2% trailing stop_loss profit:", z_twop_trail_profit)
    print("0.5% trailing stop_loss profit:", z_fivep_trail_profit)
    print("1% trailing stop_loss profit:", onep_trail_profit)
    
    print("\nPerformance against max profit\n")

    print("Actual profit/Max profit:", actual_profit/max_profit)
    print("Buy and Hold profit/Max profit:", bnh_profit/max_profit)
    print("Standard deviation profit/Max profit:", stdDev_profit/max_profit)
    print("0.1% fixed stop_loss profit/Max profit:", z_onep_fix_profit/max_profit)
    print("0.2% fixed stop_loss profit/Max profit:", z_twop_fix_profit/max_profit)
    print("0.5% fixed stop_loss profit/Max profit:", z_fivep_fix_profit/max_profit)
    print("1% fixed stop_loss profit/Max profit:", onep_fix_profit/max_profit)
    print("0.1% trailing stop_loss profit/Max profit:", z_onep_trail_profit/max_profit)
    print("0.2% trailing stop_loss profit/Max profit:", z_twop_trail_profit/max_profit)
    print("0.5% trailing stop_loss profit/Max profit:", z_fivep_trail_profit/max_profit)
    print("1% trailing stop_loss profit/Max profit:", onep_trail_profit/max_profit)

    print("\nProfit against investment\n")
    print("Maximum profit/investment:", max_profit/total_investment)
    print("Actual profit/investment:", actual_profit/total_investment)
    print("Buy and Hold profit/investment:", bnh_profit/total_investment)
    print("Standard deviatio profit/investment:", stdDev_profit/total_investment)
    print("0.1% fixed stop_loss profit/investment:", z_onep_fix_profit/total_investment)
    print("0.2% fixed stop_loss profit/investment:", z_twop_fix_profit/total_investment)
    print("0.5% fixed stop_loss profit/investment:", z_fivep_fix_profit/total_investment)
    print("1% fixed stop_loss profit/investment:", onep_fix_profit/total_investment)
    print("0.1% trailing stop_loss profit/investment:", z_onep_trail_profit/total_investment)
    print("0.2% trailing stop_loss profit/investment:", z_twop_trail_profit/total_investment)
    print("0.5% trailing stop_loss profit/investment:", z_fivep_trail_profit/total_investment)
    print("1% trailing stop_loss profit/investment:", onep_trail_profit/total_investment)


# if __name__ == "__main__":
#     df = pd.read_csv("")

#     prepare_stop_loss_df(df)
