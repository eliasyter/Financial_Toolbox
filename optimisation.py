import yfinance as yf
import numpy as np
import numpy.linalg as npl
import pandas as pd



def Mean_Variance_No_Constraints_Naive(a,r_p,period='1mo', interval='1d', start=None, end=None):
    #import Tickers object
    tickers = yf.Tickers(a)
    names = "".join(a)
    print(names)

    #extract pricedata
    df = tickers.history(period=period,interval=interval,start=None,end=None)["Close"]
    
    #convert to numpy
    data = df.to_numpy()
    data = data[:-1]

    data = data[:-1,:]/data[1:,:] - 1
    data = np.log(1 + data) #log returns
    del df
    del tickers

    N = np.shape(data)[1]

    r = np.mean(data,axis=0) * 252

    cov = np.cov(data.T) * 252
    cov_1 = npl.solve(cov,np.identity(N))
    print(f"cov_1 shape {np.shape(cov_1)}")
    A = np.ones((1,N)) @ cov_1 @ r
    B = r.T @ cov_1 @ r
    C = np.ones(N).T @ cov_1 @ np.ones((N,1))
    D = B*C - A**2
    lam = (C*r_p - A)/D
    gam = (B - A*r_p)/D
    
    return lam*cov_1@r + gam*cov_1@np.ones(N)

    
    