


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
    income_stmt = ticker.income_stmt
    timestamps = income_stmt.columns.tolist()
    dates=[]
    pe_date=[]
    for rapport_date in timestamps:
        stock_price = ticker.history(start=pd.Timestamp(rapport_date) - pd.Timedelta(days=7), 
                                  end=pd.Timestamp(rapport_date) + pd.Timedelta(days=7))['Close']
        temp_price=0
        for i,price in enumerate(stock_price):
            temp_price+=price
        avrage_price= temp_price/(i+1)
        profit=income_stmt[rapport_date]["Gross Profit"]
        outstanding_shares=ticker.info.get('sharesOutstanding')
        pe=PE(avrage_price,profit,outstanding_shares)
        dates.append(rapport_date.year)
        pe_date.append(pe)

    plt.plot(dates, pe_date, marker='o', linestyle='-', color='b')

    
    plt.title(f"PE_Over_Time for {ticker_symbol}")
    plt.xlabel("Year")
    plt.ylabel("PE")
    plt.show()
        
    

    

def PE(price:int, earnings_12months:int, amount_stocks: int)->int: 
    pe = (price*amount_stocks)/earnings_12months
    return pe



















