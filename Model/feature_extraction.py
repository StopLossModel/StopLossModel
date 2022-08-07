from operator import pos
import pandas as pd
import numpy as np
import math
from pandas.core.frame import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

######################### TA Imports #########################
#VolumeIndicators
from ta.volume import (
    AccDistIndexIndicator
)
# Momentum indicators
from ta.momentum import (
    RSIIndicator,
    StochasticOscillator,
    WilliamsRIndicator,
)
# Tresnd indicators
from ta.trend import (
    MACD,
    CCIIndicator,
    SMAIndicator,
)

######################### TA Functions #########################
# Add volume indicators
def add_volume_ta(
        df: pd.DataFrame,
        high: str,
        low: str,
        close: str,
        volume: str,
        fillna: bool = False,
        colprefix: str = "",
) -> pd.DataFrame:
    # Accumulation Distribution Index
    df[f"{colprefix}volume_adi"] = AccDistIndexIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
    ).acc_dist_index()
    return df

# Add momentum indicators
def add_momentum_ta(
        df: pd.DataFrame,
        high: str,
        low: str,
        close: str,
        volume: str,
        fillna: bool = False,
        colprefix: str = "",
) -> pd.DataFrame:
    # Relative Strength Index (RSI)
    df[f"{colprefix}momentum_rsi"] = RSIIndicator(
                                        close=df[close], window=14, fillna=fillna
                                    ).rsi()

    # Williams R Indicator
    df[f"{colprefix}momentum_wr"] = WilliamsRIndicator(
                                        high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna
                                    ).williams_r()
    return df

# Add trend indicators
def add_trend_ta(
        df: pd.DataFrame,
        high: str,
        low: str,
        close: str,
        fillna: bool = False,
        colprefix: str = "",
) -> pd.DataFrame:
     # MACD
    indicator_macd = MACD(
                        close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
                    )
    df[f"{colprefix}trend_macd"] = indicator_macd.macd()

    # SMAs
    df[f"{colprefix}trend_sma_fast"] = SMAIndicator(
        close=df[close], window=10, fillna=fillna
    ).sma_indicator()

    # CCI Indicator
    df[f"{colprefix}trend_cci"] = CCIIndicator(
        high=df[high],
        low=df[low],
        close=df[close],
        window=20,
        constant=0.015,
        fillna=fillna,
    ).cci()
    return df

######################### Manual Implementations #########################

#weighted moving average
def get_wma(df:pd.DataFrame, close:str, period) -> pd.DataFrame:
    weights = np.arange(1,period+1)
    df["wma"] = df[close].rolling(period).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    return df

#alan hull indicator
def add_alan_hull(df: pd.DataFrame, close: str, period=10) -> pd.DataFrame:
    # HMA[i] = MA( (2*MA(input, period/2) – MA(input, period)), SQRT(period))
    sqrt_prd = int(math.sqrt(period))
    weights = np.arange(1,period+1)
    weights_hf = np.arange(1, (period/2)+1)
    weights_sqrt = np.arange(1, sqrt_prd+1)

    df["period_ma"] = df[close].rolling(period).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    df["period_hf_ma"] = df[close].rolling(int(period/2)).apply(lambda prices: np.dot(prices, weights_hf)/weights_hf.sum(), raw=True)
    df["period_hf_ma"] = df["period_hf_ma"].multiply(2)
    df["subs_ma"] = df["period_hf_ma"].subtract(df["period_ma"])

    df["alan_h_ma"] = df["subs_ma"].rolling(sqrt_prd).apply(lambda prices: np.dot(prices, weights_sqrt)/weights_sqrt.sum(), raw=True)

    return df

#Momentum
def get_momentum(df:pd.DataFrame, close:str, n=9)->pd.DataFrame:
    df["momentum"] = df[close].diff(n)  # calculates diff in most recent closing price from closing price n periods ago
    return df

#stochasticK
def get_stochasticK(df:pd.DataFrame, low:str, high:str, close:str, k=10)->pd.DataFrame:
    # Set minimum low and maximum high of the k stoch
    low_min = df[low].rolling(window=k).min()
    high_max = df[high].rolling(window=k).max()

    # Fast Stochastic
    k_fast = 100 * (df[close] - low_min) / (high_max - low_min)
    df["stoch_k"] = k_fast
    return df

#stochasticD
def get_stochasticD(df:pd.DataFrame, low:str, high:str, close:str, k=10, d=3):
    # Set minimum low and maximum high of the k stoch
    low_min = df[low].rolling(window=k).min()
    high_max = df[high].rolling(window=k).max()

    # slow Stochastic
    k_fast = 100 * (df[close] - low_min) / (high_max - low_min)
    d_slow = k_fast.rolling(window=d).mean()
    df["stoch_d"] = d_slow
    return df

######################### PCA Function #########################
def pca(df: pd.DataFrame, features: list, target: str, n_comps=2):
    X = df[features].values
    y = df[target].values

    # Standardize the features
    st_scaler = StandardScaler()
    X = st_scaler.fit_transform(X)
    # Instantiate PCA
    pca = PCA(n_comps)

    # Fit PCA to features
    principalComponents = pca.fit_transform(X)

    dummy_data = np.zeros((principalComponents.shape[0], X.shape[-1] - principalComponents.shape[-1]))
    comps_dummy = np.append(principalComponents, dummy_data, axis=1)

    scaled_back = st_scaler.inverse_transform(comps_dummy)

    # Calculate the variance explained by priciple components
    print('Variance of each component:', pca.explained_variance_ratio_)
    print('\n Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_)) * 100, 2))

    return scaled_back[:, :principalComponents.shape[-1]], st_scaler, pca

def prepare_df(data):
    # Add technical indicators
    print("Adding TAs")
    add_volume_ta(data, "High", "Low", "Close", "Volume", False, "")
    add_momentum_ta(data, "High", "Low", "Close", "Volume", False, "")
    add_trend_ta(data, "High", "Low", "Close", False, "")

    get_wma(data, "Close", 10)
    get_momentum(data, "Close", 9)
    get_stochasticK(data, "Low", "High", "Close", 10)
    get_stochasticD(data, "Low", "High", "Close", 10, 3)

    data.dropna(inplace=True)
    # print(data.head(10).to_string())
    
    return data

def perform_pca(data, start_col_index=5):
    # Perform PCA
    features = data.columns[start_col_index:]
    prev_features = data.columns[:start_col_index]
    pcs, pc_scaler, pc_model = pca(data, features, "Close", 0.8)

    # print(pcs)

    # append PCs to data
    pc_heads = ["PC"+str(i+1) for i in range(0, pcs.shape[-1])]
    # print(pc_heads)

    data[pc_heads] = pcs

    data_headers = ["Close", "Open", "High", "Low", "Volume"] + pc_heads

    data = data[data_headers]

    return data, pc_scaler, pc_model

def fit_pca(data, pc_model, pc_scaler, start_col_index=5):
    features = data.columns[start_col_index:]
    prev_features = data.columns[:start_col_index]
    ft_vals = data[features].values
    ft_scaled = pc_scaler.transform(ft_vals)
    pcs = pc_model.transform(ft_scaled)
    dummy_data = np.zeros((pcs.shape[0], ft_scaled.shape[-1] - pcs.shape[-1]))
    comps_dummy = np.append(pcs, dummy_data, axis=1)
    scaled_back = pc_scaler.inverse_transform(comps_dummy)
    pcs = scaled_back[:, :pcs.shape[-1]]

    pc_heads = ["PC"+str(i+1) for i in range(0, pcs.shape[-1])]
    # print(pc_heads)

    data[pc_heads] = pcs
    data_headers = ["Close", "Open", "High", "Low", "Volume"] + pc_heads
    data = data[data_headers]

    return data

def process_indicators(df:pd.DataFrame, close:str, sma:str, wma:str, stK:str, stD:str, will:str, macd:str, rsi:str, cci:str, ad:str, momentum:str)->pd.DataFrame:
    pr_sma = []
    pr_wma = []
    pr_stK = []
    pr_stD = []
    pr_will = []
    pr_macd = []
    pr_rsi = []
    pr_cci = []
    pr_ad = []
    pr_moment = []
    prev_row = df.iloc[0]
    for i in range(0, len(df.index)):
        row = df.iloc[i]
        prev_close = prev_row[close]
        r_close = row[close]
        # SMA
        if r_close > row[sma]:
            pr_sma.append(1)
        else:
            pr_sma.append(-1)

        # WMA
        if r_close > row[wma]:
            pr_wma.append(1)
        else:
            pr_wma.append(-1)

        # stochasticK
        if row[stK] > prev_row[stK]:
            pr_stK.append(1)
        else:
            pr_stK.append(-1)
        
        # stochasticD
        if row[stD] > prev_row[stD]:
            pr_stD.append(1)
        else:
            pr_stD.append(-1)
        
        # William's oscillator
        if row[will] > prev_row[will]:
            pr_will.append(1)
        else:
            pr_will.append(-1)
        
        # MACD
        if row[macd] > prev_row[macd]:
            pr_macd.append(1)
        else:
            pr_macd.append(-1)
        
        # RSI
        if row[rsi] > 70:
            pr_rsi.append(-1)
        elif row[rsi] < 30:
            pr_rsi.append(1)
        else:
            if row[rsi] > prev_row[rsi]:
                pr_rsi.append(1)
            else:
                pr_rsi.append(-1)
        
        # cci
        if row[cci] > 200:
            pr_cci.append(-1)
        elif row[cci] < 200:
            pr_cci.append(1)
        else:
            if row[cci] > prev_row[cci]:
                pr_cci.append(1)
            else:
                pr_cci.append(-1)
        
        # A/D oscillator
        if row[ad] > prev_row[ad]:
            pr_ad.append(1)
        else:
            pr_ad.append(-1)
        
        # Momentum
        if row[momentum] >= 0:
            pr_moment.append(1)
        else:
            pr_moment.append(-1)
    
        prev_row = row
    
    df["pr_sma"] = pr_sma
    df["pr_wma"] = pr_wma
    df["pr_stK"] = pr_stK
    df["pr_stD"] = pr_stD
    df["pr_will"] = pr_will
    df["pr_macd"] = pr_macd
    df["pr_rsi"] = pr_rsi
    df["pr_cci"] = pr_cci
    df["pr_ad"] = pr_ad
    df["pr_moment"] = pr_moment

    # df = df.drop([sma, wma, stK, stD, will, macd, rsi, cci, ad, momentum], axis=1)

    return df

def get_trend_change(df:pd.DataFrame, close:str):
    print("Adding trend")
    df["trend"] = df[close].diff()
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df

        

