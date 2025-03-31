import yfinance as yf
import numpy as np
import numpy.linalg as npl
import pandas as pd
import get_data

def Mean_Variance(log_returns, mu_p):
    N     = log_returns.shape[1]
    r     = np.mean(log_returns,axis=0) * 252
    cov   = np.cov(log_returns.T) * 252
    cov_1 = npl.solve(cov,np.identity(N))
    A     = np.ones((1,N)) @ cov_1 @ r
    B     = r.T @ cov_1 @ r
    C     = np.ones(N).T @ cov_1 @ np.ones((N,1))
    D     = B*C - A**2
    lam   = (C*mu_p - A)/D
    gam   = (B - A*mu_p)/D
    w     = lam*cov_1@r + gam*cov_1@np.ones(N) 
    return w


def Mean_Variance_No_Constraints_Naive(a, r_p, period='1mo', interval='1d', start=None, end=None):
    #import Tickers object
    tickers = yf.Tickers(a)
    names = a.split(" ")

    #extract pricedata
    df = tickers.history(period=period,interval=interval,start=start,end=end)["Close"]
    
    #convert to numpy
    data = df.to_numpy()
    data = data[:-1]

    data = data[:-1,:]/data[1:,:] - 1
    data = np.log(1 + data) #log returns
    del df
    del tickers

    N = np.shape(data)[1]

    r = np.mean(data,axis=0) * 252
    mu_p = np.log(r_p+1)

    cov = np.cov(data.T) * 252
    cov_1 = npl.solve(cov,np.identity(N))
    print(f"cov_1 shape {np.shape(cov_1)}")
    A     = np.ones((1,N)) @ cov_1 @ r
    B     = r.T @ cov_1 @ r
    C     = np.ones(N).T @ cov_1 @ np.ones((N,1))
    D     = B*C - A**2
    lam   = (C*mu_p - A)/D
    gam   = (B - A*mu_p)/D

    return lam*cov_1@r + gam*cov_1@np.ones(N),names

def Mean_Variance_No_Constraints_Robust(a, r_p, period='1mo', interval='1d', start=None, end=None):
    log_returns,names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end)
    mu_p = np.log(r_p+1)
    weights = Mean_Variance(log_returns, mu_p)

    return weights,names

def Mean_Variance_No_Shortselling(a, r_p, period='1mo', interval='1d', start=None, end=None):
    log_returns,names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end)
    names = np.array(names)
    mu_p = np.log(r_p+1)
    max_iter = log_returns.shape[1] # we can maximum iterate N times
    long_map = np.ones(log_returns.shape[1], dtype=bool) # to begin we consider every weight

    N     = int(log_returns.shape[1])
    cov   = np.cov(log_returns.T) * 252
    cov_1 = npl.solve(cov,np.identity(N))
    r     = np.mean(log_returns,axis=0) * 252

    print(f"r = {r}, shape = {r.shape}")
    print(f"cov_1 shape = {cov_1.shape}")
    
    for i in range(max_iter):
        
        cov_1 = cov_1 * np.outer(long_map,long_map)
        # recompute variables
        A       = np.ones((1,N)) @ cov_1 @ r
        B       = r.T @ cov_1 @ r
        C       = np.ones((1,N)) @ cov_1 @ np.ones((N,1))
        D       = B*C - A**2
        if D == 0:
            # something went wrong
            weights = np.array([])
            print("failed to find optimal pf")
            break
        lam     = (C*mu_p - A)/D
        gam     = (B - A*mu_p)/D
        weights = lam*cov_1@r + gam*cov_1@np.ones(N) 

        short_map = weights < 0
        long_map  = weights >= 0

        if short_map.sum()==0:
            # every weight is non-negative
            break
        
    return weights,names

def Mean_Variance_No_Shortselling_Robust(a, r_p, period='1mo', interval='1d', start=None, end=None):
    log_returns,names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end)
    return np.zeros(len(names)),names
    