

# ticker i string
# hente earnings 
# hent price 
# kalkuler pe 
# returnerer pandas dataframe 


# plot

import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd 


import yfinance as yf

def calculate_PE(ticker_symbol="TSLA"):    
    ticker = yf.Ticker(ticker_symbol)
    #pd.set_option('display.max_rows', None)
    income_stmt = ticker.income_stmt
    timestamps = income_stmt.columns.tolist()
    stock_price = ticker.history(start=timestamps[-1], end=timestamps[0])['Close']
    
    
    income_stmt = ticker.income_stmt
    timestamps = income_stmt.columns.tolist()
    outstanding_shares=ticker.info.get('sharesOutstanding')
    historical_PE={}
    i=0
    profit=income_stmt[timestamps[i]]["Net Income"]
    timestamps.reverse()
    for price_date in dict(stock_price):
        if price_date.tz_localize(None)>timestamps[i]:
            #passer på at jeg ikke går over lengden av listen
            if len(timestamps)<i+1:
                i+=1
                profit=income_stmt[timestamps[i]]["Net Income"]
        historical_PE[price_date]=PE(price=stock_price[price_date], earnings_12months=profit, amount_stocks=outstanding_shares)
    
    
    data_frame=pd.DataFrame.from_dict(historical_PE, orient='index', columns=['PE'])

    
    
    return data_frame 
        
        
    
def print_PE(data_frame):
    plt.figure(figsize=(10, 5))
    plt.plot(data_frame.index,data_frame['PE'], marker='', linestyle='-', color='b')
        

    plt.title(f"PE_Over_Time")
    plt.xlabel("Year")
    plt.ylabel("PE")
    plt.show()

    

def PE(price:int, earnings_12months:int, amount_stocks: int)->int: 
    pe = (price*amount_stocks)/earnings_12months
    return pe



















