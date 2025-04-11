import yfinance as yf
from get_data import Get_Normalized_Data
import matplotlib.pyplot as plt


#foreløbig bare for testing
import sys
import io


class Stock():
    def __init__(self,ticker_str:str, max_price_data):
        self.ticker_str=ticker_str
        self.max_price_data=max_price_data
        
    
    def get_price(self,start_date:int, end_date:int):
        new_price_intervall= self.max_price_data[(self.max_price_data.index >= start_date) & (self.max_price_data.index <= end_date)]
        return new_price_intervall



class Portfolio():
    #tickers er en liste med tuples der rekkefølgen er [(ticker,waight,amount)]
    #waight skal være på prosent formen 0.45 eks 

    #a is a string of tickers with spases between them. 
    #exp: "TSLA ORK.OL"
    def __init__(self,data:list[(str,int,int)], a:str):
        #forsikrer seg at summen av vektene dine blir til 
        
        total_prosent = sum(tup[1] for tup in data)
        assert total_prosent ==1, "You have not written 100% of your portfolio"
        self.portfolio_info={}

        a=""
        for info in data:
            a+=" "+info[0]
        prices=Get_Normalized_Data(a,period="max")[0]
        for info in data:
            Stock(info[0], prices[info[0]])

        
       
            
        
        
    
    def simulate_portfolio(self, start:str, end:str, start_amount: int):
        
        stocks={}
        for stock in self.portfolio_info:
            stocks[stock]=stock.get_price(start,end)["Close"]
    import sys
import io
def plot_test_data(a):
    #dette forhindrer funskjonen get_normalized data å printe til konsollen
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    data = Get_Normalized_Data(a, period="max")[0]
    sys.stdout = old_stdout
    

   
    #for day_indeks in range(data.shape[0]):


    print(data["TSLA"])
    data.plot()
    plt.show() 




    

        
        
        

            






