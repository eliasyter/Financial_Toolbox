import yfinance as yf
import numpy as np
import numpy.linalg as npl
import pandas as pd
import get_data

def Mean_Variance(mu_p, log_returns):
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
    return w,cov


def Mean_Variance_No_Constraints_Naive(a, r_p, period='1mo', interval='1d', start=None, end=None, verbose=True):
    #import Tickers object
    tickers = yf.Tickers(a)
    names = a.split(" ")

    #extract pricedata
    data_df = tickers.history(period=period,interval=interval,start=start,end=end)["Close"]
    
    #convert to numpy
    data = data_df.to_numpy()
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
    A     = np.ones((1,N)) @ cov_1 @ r
    B     = r.T @ cov_1 @ r
    C     = np.ones(N).T @ cov_1 @ np.ones((N,1))
    D     = B*C - A**2
    lam   = (C*mu_p - A)/D
    gam   = (B - A*mu_p)/D

    if verbose:
        print(f"cov_1 shape {np.shape(cov_1)}")
    
    weights = lam*cov_1@r + gam*cov_1@np.ones(N),names
    return weights

def Mean_Variance_No_Constraints_Robust(a, r_p, period='1mo', interval='1d', start=None, end=None, verbose=True):
    df = pd.DataFrame(columns=["Vol", "Ret", "Ret/Vol", "Comp"])
    log_returns,names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end,verbose=verbose)
    mu_p = np.log(r_p+1)
    weights,cov = Mean_Variance(mu_p, log_returns)
    # Final portfolio calculation
    R_yearly = Yearly_Arth_Returns(a, period=period, interval=interval, 
                                 start=start, end=end, verbose=verbose).values()
    R_yearly = np.array(list(R_yearly))
    
    # Ensure proper array shapes
    weights = weights.reshape(-1, 1)  # Column vector
    Vol = np.sqrt(weights.T @ cov @ weights).item()
    Ret = (weights.T @ R_yearly).item()
    
    # Create composition dict with proper scalar values
    comp_dict = {name: float(w) for name, w in zip(names, weights.flatten())}
    
    # Create new row as Series with explicit dtype conversions
    new_row = pd.Series({
        'Vol': float(Vol),
        'Ret': float(Ret),
        'Ret/Vol': float(Ret/Vol),
        'Comp': comp_dict
    })
    
    # Append with concat instead of loc for better type preservation
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    
    # Force dtypes
    df = df.astype({
        'Vol': 'float64',
        'Ret': 'float64',
        'Ret/Vol': 'float64',
        'Comp': 'object'
    })
    
    return df

def Yearly_Log_Returns(a, period='1mo', interval='1d', start=None, end=None, verbose=True):
    log_returns, names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end,verbose=verbose)
    total_log_returns = log_returns.mean(axis=0)
    if verbose:
        print("mean of returns")
        print("--------------------")
        print(total_log_returns)
        print("")
    total_log_returns = total_log_returns*252
    return dict(zip(names,total_log_returns))
    
def Yearly_Arth_Returns(a, period='1mo', interval='1d', start=None, end=None, verbose=True):
    prices, names = get_data.Get_Normalized_Data(a,period=period,interval=interval,start=start,end=end,verbose=verbose)
    n = prices.shape[0]
    total_arth_returns = prices.iloc[n-1]/prices.iloc[0]
    if verbose:
        print("total returns over period")
        print("---------------------------------")
        print(total_arth_returns-1)
        print("")
    total_arth_returns = total_arth_returns**(252/n)
    total_arth_returns = total_arth_returns -1
    return dict(zip(names,total_arth_returns))

def Mean_Variance_No_Shortselling(a, r_p, period='1mo', interval='1d', start=None, end=None, verbose=True):
    df = pd.DataFrame(columns=["Vol", "Ret", "Ret/Vol", "Comp"])
    
    # Data collection
    log_returns, names = get_data.Get_Normalized_Log_Returns(a, period=period, interval=interval, 
                                                            start=start, end=end, verbose=verbose)
    names = np.array(names)
    mu_p = np.log(r_p+1)
    max_iter = log_returns.shape[1]
    long_map = np.ones(log_returns.shape[1], dtype=bool)
    N = int(log_returns.shape[1])
    
    # Portfolio calculations
    cov = np.cov(log_returns.T) * 252
    cov_1 = npl.solve(cov, np.identity(N))
    r = np.mean(log_returns, axis=0) * 252
    
    if verbose:
        print(f"r = {r}, shape = {r.shape}")
        print(f"cov_1 shape = {cov_1.shape}")

    for i in range(max_iter):
        cov_1 = cov_1 * np.outer(long_map, long_map)
        
        # Portfolio optimization math
        A = np.ones((1,N)) @ cov_1 @ r
        B = r.T @ cov_1 @ r
        C = np.ones((1,N)) @ cov_1 @ np.ones((N,1))
        D = B*C - A**2
        
        if D == 0:
            print("Failed to find optimal pf")
            return df  # Return empty DF instead of breaking
            
        lam = (C*mu_p - A)/D
        gam = (B - A*mu_p)/D
        weights = lam*cov_1@r + gam*cov_1@np.ones(N)
        weights = weights.flatten()  # Ensure 1D array

        # Long/short check
        short_map = weights < 0
        long_map = weights >= 0

        if short_map.sum() == 0:
            break
            
    # Final portfolio calculation
    R_yearly = Yearly_Arth_Returns(a, period=period, interval=interval, 
                                 start=start, end=end, verbose=verbose).values()
    R_yearly = np.array(list(R_yearly))
    
    # Ensure proper array shapes
    weights = weights.reshape(-1, 1)  # Column vector
    Vol = np.sqrt(weights.T @ cov @ weights).item()
    Ret = (weights.T @ R_yearly).item()
    
    # Create composition dict with proper scalar values
    comp_dict = {name: float(w) for name, w in zip(names, weights.flatten())}
    
    # Create new row as Series with explicit dtype conversions
    new_row = pd.Series({
        'Vol': float(Vol),
        'Ret': float(Ret),
        'Ret/Vol': float(Ret/Vol),
        'Comp': comp_dict
    })
    
    # Append with concat instead of loc for better type preservation
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    
    # Force dtypes
    df = df.astype({
        'Vol': 'float64',
        'Ret': 'float64',
        'Ret/Vol': 'float64',
        'Comp': 'object'
    })

    return df



def Mean_Variance_No_Shortselling_Robust(a, r_p, period='1mo', interval='1d', start=None, end=None, verbose=True):
    log_returns,names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end,verbose=verbose)
    return dict(zip(names,np.zeros(len(names))))

def Efficiency_Frontier_No_Constraints(a, min_ret, max_ret, prec=0.1, period='1mo', interval='1d', start=None, end=None, verbose=True):
    EF_df = pd.DataFrame(columns=["Ret", "Vol", "Ret/Vol", "Comp"])
    log_returns, names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end,verbose=verbose)
    N = len(names)
    cov   = np.cov(log_returns.T) * 252
    cov_1 = npl.solve(cov, np.identity(N))
    R = np.arange(min_ret, max_ret+prec,prec)
    R_log = np.log(R + 1)
    R_yearly = Yearly_Arth_Returns(a,period=period,interval=interval,start=start,end=end,verbose=verbose).values()
    R_yearly = np.array(list(R_yearly)).reshape((N,1))
    
    if verbose:
        print("---------------")
        print(f"R    : {R}")
        print(f"R log: {R_log}")
        print(f"shape of cov_1: {cov_1.shape}")
        print("")
    
    for i in range(len(R_log)):
        mu = R_log[i]
        weights,_ = Mean_Variance(mu, log_returns)
        weights = weights.reshape((N,1))
        if verbose:
            print(f"on iteration {i+1}")
            print("-----------------------------")
            print(f"shape of weights: {weights.shape}")
            print("")
        Vol = (weights.T @ cov @ weights)**0.5
        Ret = weights.T @ R_yearly
        EF_df.loc[len(EF_df)] = [Ret, Vol, Ret/Vol, dict(zip(names,weights))]
        if verbose:
            print(f"finished {i+1}-th iteration")
    
    stocks_df = pd.DataFrame(columns=["Ret", "Vol", "Ret/Vol", "Comp"])
    # add stocks for plotting
    for i in range(N):
        temp_weights = np.zeros(N)
        temp_weights[i] = 1
        stocks_df.loc[names[i]] = [R_yearly[i].item(), cov[i,i], R_yearly[i]/cov[i,i], dict(zip(names,temp_weights))]
        
    return EF_df, stocks_df
    
    
    