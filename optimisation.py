import yfinance as yf
import numpy as np
import numpy.linalg as npl
import pandas as pd
import get_data


#---------------- Helpers ----------------# 

def Mean_Variance_Kernel(exp_ret, returns):
    n,p   = returns.shape
    r     = np.mean(returns, axis=0) * 252 #returns are not compounding
    cov   = np.cov(returns.T) * 252
    cov_1 = npl.solve(cov, np.identity(p))
    A     = np.ones(p).T @ cov_1 @ r
    B     = r.T @ cov_1 @ r
    C     = np.ones(p).T @ cov_1 @ np.ones((p,1))
    D     = B*C - A**2
    lam   = (C*exp_ret - A)/D
    gam   = (B - A*exp_ret)/D
    w     = lam*cov_1 @ r + gam*cov_1 @ np.ones(p)
    return w, cov, r

def Mean_Variance_No_Shortselling_Kernel(exp_ret, returns, long_map):
    n,p   = returns.shape
    r     = np.mean(returns, axis=0) * 252
    cov   = np.cov(returns.T) * 252
    cov_1 = npl.solve(cov, np.identity(p))
    cov_1 = cov_1 * np.outer(long_map, long_map)
    A     = np.ones(p).T @ cov_1 @ r
    B     = r.T @ cov_1 @ r
    C     = np.ones(p).T @ cov_1 @ np.ones(p)
    D     = B*C - A**2
    lam   = (C*exp_ret - A)/D
    gam   = (B - A*exp_ret)/D
    w     = lam*cov_1 @ r + gam*cov_1 @ np.ones(p)
    return w, cov, r

def Yearly_Log_Returns(a, period='1mo', interval='1d', start=None, end=None, verbose=True):
    log_returns, names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end,verbose=verbose)
    total_log_returns = log_returns.mean(axis=0)
    if verbose:
        print("mean of returns")
        print("--------------------")
        print(total_log_returns)
        print("")
    total_log_returns = total_log_returns*252
    total_log_returns.rename("Ret", inplace=True)
    return total_log_returns
    
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
    total_arth_returns.rename("Ret", inplace=True)
    return total_arth_returns

#---------------- Optimisers ----------------# 

def Mean_Variance_No_Constraints(a, r_p, period='1mo', interval='1d', start=None, end=None, verbose=True):
    df = pd.DataFrame(columns=["Ret", "Vol", "Ret/Vol", "Comp"])
    df = df.astype({"Ret": 'float64', "Vol": 'float64', "Ret/Vol": 'float64', "Comp": 'object'})
    log_returns,names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end,verbose=verbose)
    weights,cov,y_returns = Mean_Variance_Kernel(r_p, log_returns)
    df.loc[0] = [y_returns @ weights.T,
                 weights.T @ cov @ weights, 0., {name: w for name,w in zip(names, weights)}]
    df.iloc[0,2] = df.iloc[0,0]/df.iloc[0,1]
    return df

def Mean_Variance_No_Shortselling(a, r_p, period='1mo', interval='1d', start=None, end=None, verbose=True):
    df = pd.DataFrame(columns=["Ret", "Vol", "Ret/Vol", "Comp"])
    df = df.astype({"Ret": 'float64', "Vol": 'float64', "Ret/Vol": 'float64', "Comp": 'object'})
    log_returns,names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end,verbose=False)
    n,p = log_returns.shape
    cov = np.cov(log_returns.T) * 252
    long_map = np.ones(p, dtype=bool)

    for i in range(n):
        weights, _, y_returns = Mean_Variance_No_Shortselling_Kernel(r_p, log_returns, long_map)
        long_map = weights > 0
        short_map = weights < 0
        if verbose:
            np.set_printoptions(precision=3)
            print(f"Current iteration: {i+1}, weights = {weights}")
            print(f"cov matrix for next iteration: \n {cov*np.outer(long_map, long_map)} \n")
        if short_map.sum()==0:
            #we finished removing short-positions
            break
    df.loc[0] = [y_returns @ weights.T,
                 weights.T @ cov @ weights, 0., {name: w for name,w in zip(names, weights)}]
    df.iloc[0,2] = df.iloc[0,0]/df.iloc[0,1]
    return df

def Mean_Variance_Sparse(a, r_p, gamma, period='1mo', interval='1d', start=None, end=None, verbose=True):
    df = pd.DataFrame(columns=["Ret", "Vol", "Ret/Vol", "Comp"])
    df = df.astype({"Ret": 'float64', "Vol": 'float64', "Ret/Vol": 'float64', "Comp": 'object'})
    log_returns,names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end,verbose=False)
    n,p = log_returns.shape
    #implement proximal gradient - compromise: few assets, low variance

    return None

#---------------- Efficient Frontier ----------------# 
def Efficient_Frontier_No_Constraints(a, min_ret, max_ret, prec=0.1, period='1mo', interval='1d', start=None, end=None, verbose=True):
    df = pd.DataFrame(columns=["Ret", "Vol", "Ret/Vol", "Comp"])
    df = df.astype({"Ret": 'float64', "Vol": 'float64', "Ret/Vol": 'float64', "Comp": 'object'})
    log_returns, names = get_data.Get_Normalized_Log_Returns(a,period=period,interval=interval,start=start,end=end,verbose=False)
    n, p = log_returns.shape
    return_levels = np.arange(min_ret, max_ret+prec, prec)
    for i, r_p in enumerate(return_levels):
        weights, cov, y_returns = Mean_Variance_Kernel(r_p, log_returns)
        if verbose:
            np.set_printoptions(precision=3)
            print(f"Current iteration: {i+1} (exp_ret : {r_p:.2%}), weights = {weights}")
            print(f"cov matrix: \n {cov} \n")
        df.loc[len(df)] = [y_returns @ weights.T,
                           weights.T @ cov @ weights, 0., {name: w for name,w in zip(names, weights)}] #creates new row
        df.iloc[i,2] = df.iloc[i,0]/df.iloc[i,1]
        if verbose:
            print(f"finished the {i+1}th iteration")
            print(f"added to df: {df.loc[0]} \n")
    
    # add stocks for plotting
    for i, name in enumerate(names):
        temp_weights = np.zeros(p)
        temp_weights[i] = 1.
        df.loc[len(df)] = [y_returns.iloc[i], cov[i,i], y_returns.iloc[i]/cov[i,i], {name: w for name,w in zip(names, temp_weights)}]
    return df
    
    
    