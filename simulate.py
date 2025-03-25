import yfinance as yf


class Stock():
    def __init__(self,ticker_str:str):
        self.ticker_str=ticker_str
        self.max_price_data=yf.Ticker(ticker_str).history(period="max")
        
    
    def get_price(self,start_date:int, end_date:int):
        new_price_intervall= self.max_price_data[(self.max_price_data.index >= start_date) & (self.max_price_data.index <= end_date)]
        return new_price_intervall



class Portfolio():
    #tickers er en liste med tuples der rekkefølgen er [(ticker,waight,amount)]
    #waight skal være på prosent formen 0.45 eks 
    def __init__(self,data:list[(str,int,int)]):
        #forsikrer seg at summen av vektene dine blir til 1 
        total_prosent = sum(tup[1] for tup in data)
        assert total_prosent ==1, "You have not written 100% of your portfolio"
        self.portfolio_info={}

        #this should be a list of the first data point that we have of a stock price
        #this is made so i can calulate the first data point that i have
        first_data_point=[] 
        for ticker in data:
            current_stock=Stock(ticker[0])
            first_data_point.append(current_stock.max_price_data.iloc[0].name)
            self.portfolio_info[current_stock]={"waight":ticker[1],
                                                "amount":ticker[2]}
            
        #this will used to never go over the point that i have data on all stocks
        #litt spess at det er max her men det funker så la det være
        #den henter den aller føste datoen der man har data for alle stock prisene. 
        self.first_data_point= max(first_data_point)
        
        
    
    def simulate_portfolio(self, start:str, end:str, start_amount: int):
        
        stocks={}
        for stock in self.portfolio_info:
            stocks[stock]=stock.get_price(start,end)["Close"]
        
        

            
            
    
        
        




















