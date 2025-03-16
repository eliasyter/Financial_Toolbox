import yfinance as yf
import numpy as np
import numpy.linalg as npl
import pandas as pd

def Get_Normalized_Data(a, period='1mo', interval='1d', start=None, end=None):
    names = a.split(" ")
    
    #identify different currencies
    currencies = {}
    
    for name in names:
        currencies[name] = yf.Ticker(name).info.get("currency")

    #import currency time series
    cst = list(set(currencies.values()))
    try:
        cst.remove("USD")
    except ValueError:
        pass
    tmp = [c+"=X" for c in cst]
    tmp = " ".join(tmp)

    currencies_tickers = yf.Tickers(tmp)
    currencies_time_series = currencies_tickers.history(period=period,interval=interval,start=start,end=end)["Close"]
    currencies_time_series.columns = cst
    
    print("Currency time series")
    print("------------------------------")
    print(currencies_time_series)
    print("")

    print("Currencies dictionary")
    print("------------------------------")
    print(currencies)
    print("")

    df = yf.Tickers(a).history(period=period,interval=interval,start=start,end=end)["Close"]
    for name in currencies.keys():
        if currencies[name]!="USD":
            df[name] = df[name]/currencies_time_series[currencies[name]]

    print("Data, in USD")
    print("------------------------------")
    print(df)
    print("")
    print(f"Row count before cleaning: {df.count()}")
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"Row count after cleaning: {df.count()}")
    print("")
    
    return df,names

def Get_Normalized_Returns(a, period='1mo', interval='1d', start=None, end=None):
    data,names = Get_Normalized_Data(a,period=period,interval=interval,start=start,end=end)
    returns = data.pct_change(periods=-1).iloc[:-1]
    return returns,names

def Get_Normalized_Log_Returns(a, period='1mo', interval='1d', start=None, end=None):
    data,names = Get_Normalized_Data(a,period=period,interval=interval,start=start,end=end)
    log_returns = data.pct_change(periods=-1).iloc[:-1] + 1
    log_returns = log_returns.apply(np.log)
    print("Log returns")
    print("------------------------------")
    print(log_returns)
    print("")
    return log_returns,names