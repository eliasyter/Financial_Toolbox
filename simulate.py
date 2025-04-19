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
    #data er en liste med tuples der rekkefølgen er [(ticker,waight,amount)]
    #waight skal være på prosent formen 0.45 eks 

    #a is a string of tickers with spases between them. 
    #exp: "TSLA ORK.OL"
    def __init__(self,data:list[(str,int,int)]):
        #forsikrer seg at summen av vektene dine blir til 
        
        total_prosent = sum(tup[1] for tup in data)
        assert total_prosent ==1, "You have not written 100% of your portfolio"
        self.portfolio_info={}

        a=""
        for info in data:
            a+=" "+info[0]
        a=a[1:]
        prices=Get_Normalized_Data(a,period="max")[0]
        print(prices)
        for info in data:
            new_stock=Stock(ticker_str=info[0], max_price_data=prices[info[0]])
            self.portfolio_info[new_stock]={"waight":info[1],
                                            "amount": info[2]}


        
    def simulate_portfolio(self, start:str, end:str, investing_amount:int):
        for info in self.portfolio_info:
            df=info.get_price(start,end)
            df["Change"] = df.diff().dropna()
            for i,change in enumerate(df["Change"]):
                
                percentage_change= change/(df.iloc[i+1])
                ##print(percentage_change)
                self.portfolio_info[info]["amount"]=self.portfolio_info[info]["amount"]*(1+percentage_change)
                print(self.portfolio_info[info]["amount"])


        #print(df["Change"])

            




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




    

        
        
        

            






