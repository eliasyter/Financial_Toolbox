

class Portfolio():
    def __init__(self,tickers:list[str]):
        stocks=[]
        for ticker in tickers:
            stocks.append(Stock(ticker))






class Stock():
    def __init__(self,ticker):
        self.ticker=ticker
        self.five_year_price=[]













