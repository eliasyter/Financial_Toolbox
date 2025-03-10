


# ticker i string
# hente earnings 
# hent price 
# kalkuler pe 
# returnerer pandas dataframe 


# plot


import yfinance as yf
import pandas as pd 


import yfinance as yf

# Angi ticker-symbolet
ticker_symbol = "AAPL"

# Hent aksjeinformasjon
stock = yf.Ticker(ticker_symbol)

print(stock.income_stmt["2024-09-30"]["Total Revenue"])






















