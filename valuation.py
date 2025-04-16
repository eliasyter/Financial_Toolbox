import yfinance as yf
import numpy as np
import pandas as pd
import get_data

#---------------- Helpers ----------------# 

#Different financial indicators, their yf ticker and listed currency
def Get_Financial_Indicators():
    df = pd.DataFrame([
    ["S&P 500 Index", "^GSPC", "USD"],
    ["Dow Jones Industry Average", "^DJI", "USD"],
    ["NASDAQ Composite", "^IXIC", "USD"],
    ["DAX Index", "^GDAXI", "EUR"],
    ["CAC 40 Index", "^FCHI", "EUR"],
    ["Nikkei 225", "^N225", "JPY"],
    ["FTSE 100", "^FTSE", "GBP"],
    ["S&P MID CAP 400 Index", "^MID", "USD"],
    ["S&P 600", "^SP600", "USD"],
    ["Dow Jones Composite Average", "^DJA", "USD"],
    ["NASDAQ Financial 100", "^IXF", "USD"],
    ["Rusell 2000", "^RUT", "USD"],
    ["NYSE Composite", "^NYA", "USD"],
    ["The Technology Select Sector SPDR Fund", "XLK", "USD"],
    ["The Financial Select Sector SPDR Fund", "XLF", "USD"],
    ["The Health Care Select Sector SPDR Fund", "XLV", "USD"],
    ["The Energy Select Sector SPDR Fund", "XLE", "USD"],
    ["The Industrial Select Sector SPDR Fund", "XLI", "USD"],
    ["The Consumer Discretionary Select Sector SPDR Fund", "XLY", "USD"],
    ["The Consumer Staples Select Sector SPDR Fund", "XLP", "USD"],
    ["The Materials Select Sector SPDR Fund", "XLB", "USD"],
    ["The Utilities Select Sector SPDR Fund", "XLU", "USD"],
    ["The Real Estate Select Sector SPDR Fund", "XLRE", "USD"],
    ["The Communication Services Select Sector SPDR ETF Fund", "XLC", "USD"],
    ["Oslo Børs Benchmark Index", "OSEBX.OL", "NOK"],
    ["OMX Copenhagen 25 Index", "^OMXC25", "DKK"],
    ["OMX Stockholm 30 Index", "^OMX", "SEK"],
    ["OMX Helsinki 25", "^OMXH25", "EUR"],
    ["Treasury Yield 30 Years", "^TYX", "USD"],
    ["Treasury Yield 5 Years", "^FVX", "USD"],
    ["Treasury Yield 13 Weeks", "^IRX", "USD"],
    ["Brent Crude Oil", "BZ=F", "USD"],
    ["Natural Gas", "NG=F", "USD"],
    ["Lumber Futures", "LBR=F", "USD"],
    ["Invesco DB Agriculture Fund", "DBA", "USD"],
    ["CBOE Volatility Index", "^VIX", "USD"],
    ["CBOE S&P 500 3-Month Volatility", "^VIX3M", "USD"],
    ["Gold", "GC=F", "USD"],
    ["Silver", "SI=F", "USD"],
    ["Copper", "HG=F", "USD"]], columns=["Index", "Ticker", "Currency"])
    return df

#Extracts the Damodaran table from New York University website containing country risk premiums
def Get_Damodaran():
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html"
    df = pd.read_html(url, header=0)
    df = df[-1]
    return df

#shipping
def Get_OSEBX_Shipping():
    stocks = "2020.OL ADS.OL AMSC.OL AGAS.OL ALNG.OL BRUT.OL BWLPG.OL CLCO.OL EWIND.OL EQVA.OL FLNG.OL FRO.OL GOGL.OL HAFNI.OL HAV.OL HKY.OL HSHP.OL HAUTO.OL JIN.OL"
    stocks += " KCC.OL MPCC.OL ODF.OL ODFB.OL OET.OL PHLY.OL SAGA.OL STST.OL STSU.OL SNI.OL WAVI.OL WEST.OL"
    return stocks

def Get_OSEBX_Shipping_Clean_Dirty():
    return "FRO.OL OET.OL HAFNI.OL KCC.OL ODF.OL ODFB.OL SNI.OL STST.OL"
    
#Map stocks-sectors-indicators

#---------------- Methods ----------------# 
def Multiples_Valuation(candidate, peers, verbose=True):
    peers_table  = get_data.Construct_Multiples_Table(peers, verbose=verbose)
    df           = pd.DataFrame(columns=["Price", "P/E", "EV/EBITDA", "EV/EBIT", "EV/Revenue"])
    df           = df.astype({"Price": 'float64', "P/E": 'float64', "EV/EBITDA": 'float64', "EV/EBIT": 'float64', "EV/Revenue": 'float64'})
    ticker       = yf.Ticker(candidate)
    ticker_info  = ticker.info
    price        = ticker_info["open"]
    nb_shares    = ticker.quarterly_balancesheet.loc["Ordinary Shares Number"].iloc[0]
    net_debt     = ticker.quarterly_balancesheet.loc["Total Debt"].iloc[0] - ticker.quarterly_balancesheet.loc["Cash And Cash Equivalents"].iloc[0]
    enterprise   = nb_shares*price + net_debt
    trailing_eps = ticker_info["epsTrailingTwelveMonths"]
    ebitda       = ticker.quarterly_income_stmt.loc["EBITDA"].iloc[:4].sum()
    ebit         = ticker.quarterly_income_stmt.loc["EBIT"].iloc[:4].sum()
    revenue      = ticker.quarterly_income_stmt.loc["Total Revenue"].iloc[:4].sum()
    #compute estimates
    peers_average = peers_table.mean(axis=0)
    df.loc[candidate] = [price,
                         peers_average["P/E"] * trailing_eps,
                         (peers_average["EV/EBITDA"] * ebitda - net_debt)/nb_shares,
                         (peers_average["EV/EBIT"] * ebit - net_debt)/nb_shares,
                         (peers_average["EV/Revenue"] * revenue - net_debt)/nb_shares]
    df.columns = pd.MultiIndex.from_tuples([
        ('actual', 'price'),
        ('multiples valuation', 'p/e'),
        ('multiples valuation', 'ev/ebitda'),
        ('multiples valuation', 'ev/ebit'),
        ('multiples valuation', 'ev/revenue')
    ])
    return df

    