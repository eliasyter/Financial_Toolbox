

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
    #print(income_stmt.loc['Gross Profit'])
    outstanding_shares=ticker.info.get('sharesOutstanding')
    historical_PE={}
    i=0
    profit=income_stmt[timestamps[i]]["Gross Profit"]
    timestamps.reverse()
    for price_date in dict(stock_price):
        if price_date.tz_localize(None)>timestamps[i]:
            #passer på at jeg ikke går over lengden av listen
            if len(timestamps)<i+1:
                i+=1
                profit=income_stmt[timestamps[i]]["Gross Profit"]
        historical_PE[price_date]=PE(price=stock_price[price_date], earnings_12months=profit, amount_stocks=outstanding_shares)
    
    
    data_frame=pd.DataFrame.from_dict(historical_PE, orient='index', columns=['PE'])

    
    
    return data_frame 


def calculate_Trailing_PE(ticker_symbol:str):
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
    profit=income_stmt[timestamps[i]]["Gross Profit"]
    timestamps.reverse()

    for price_date in dict(stock_price):
        start_date= price_date+pd.Timedelta(days=-365)
        trailing_12_months_price_list = stock_price[(stock_price.index >= start_date) & (stock_price.index <= price_date)]
        #print(trailing_12_months_price_list)
        
        trailing_stock_price=sum(trailing_12_months_price_list)/float(len(trailing_12_months_price_list))
        if price_date.tz_localize(None)>timestamps[i]:
            #passer på at jeg ikke går over lengden av listen
            if len(timestamps)<i+1:
                i+=1
                profit=income_stmt[timestamps[i]]["Gross Profit"]

        historical_PE[price_date]=PE(price=trailing_stock_price, earnings_12months=profit, amount_stocks=outstanding_shares)
    
    
    data_frame=pd.DataFrame.from_dict(historical_PE, orient='index', columns=['Trailing_PE'])

    
    
    return data_frame 

        
        
    
def print_PE(data_frames:list):
    plt.figure(figsize=(10, 5))
    names=["PE","Trailing_PE"]
    for i,data_frame in enumerate(data_frames):
        name=names[i]
        plt.plot(data_frame.index,data_frame[name], marker='', linestyle='-', label=name)
        

    plt.title(f"PE_Over_Time")
    plt.xlabel("Year")
    plt.ylabel("PE")
    plt.legend()  # Tilføj en legend for at vise labels
    plt.grid(True)  # Tilføj grid for bedre læsbarhed (valgfrit)
    plt.show()

    

def PE(price:int, earnings_12months:int, amount_stocks: int)->int: 
    pe = (price*amount_stocks)/earnings_12months
    return pe



















