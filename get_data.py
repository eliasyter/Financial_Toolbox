import yfinance as yf
import numpy as np
import numpy.linalg as npl
import pandas as pd

def Get_Normalized_Data(a, period='1mo', interval='1d', start=None, end=None, verbose=True):
    names = a.split(" ")
    
    #identify different currencies
    currencies = {}
    
    for name in names:
        currencies[name] = yf.Ticker(name).info.get("currency")

    if verbose:
        print("Currencies dictionary")
        print("------------------------------")
        print(currencies)
        print("")
        print(currencies.values())
        print("")
        
    #import currency time series
    tmp = []
    for currency in currencies.values():
        if currency != "USD" and currency not in tmp:
            tmp.append(currency)
    tmp = [currency + "=X" for currency in tmp]
    if verbose:
        print(f"tmp = {tmp}")
        print("")

    if tmp!=[]:
        currencies_tickers = yf.Tickers(tmp)
        currencies_time_series = currencies_tickers.history(period=period,interval=interval,start=start,end=end)["Close"]
        if verbose:    
            print("Currency time series")
            print("------------------------------")
            print(currencies_time_series)
            print("")

    

    df = yf.Tickers(a).history(period=period,interval=interval,start=start,end=end)["Close"]
    for name in currencies.keys():
        if currencies[name]!="USD":
            df[name] = df[name]/currencies_time_series[currencies[name]+"=X"]
    if verbose:
        print("Data, in USD")
        print("------------------------------")
        print(df)
        print("")
        print(f"Row count before cleaning: {df.count()}")
    df = df.dropna()
    df = df.drop_duplicates()
    if verbose:
        print(f"Row count after cleaning: {df.count()}")
        print("")
    
    return df,names

def Get_Normalized_Returns(a, period='1mo', interval='1d', start=None, end=None, verbose=True):
    data,names = Get_Normalized_Data(a,period=period,interval=interval,start=start,end=end,verbose=verbose)
    returns = data.pct_change(periods=1).iloc[:-1]
    return returns,names

def Get_Normalized_Log_Returns(a, period='1mo', interval='1d', start=None, end=None, verbose=True):
    data,names = Get_Normalized_Data(a,period=period,interval=interval,start=start,end=end,verbose=verbose)
    log_returns = data.pct_change(periods=1).iloc[:-1] + 1
    log_returns = log_returns.dropna()
    log_returns = log_returns.apply(np.log)
    if verbose:
        print("Log returns")
        print("------------------------------")
        print(log_returns)
        print("")
        print("characterisation of obtained Log returns")
        print("-----------------------------------------------------")
        print(log_returns.describe())
        print("")
    return log_returns,names