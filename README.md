TODO:

* optimisation.py: 
- add return output showing actual start and actual end
- implement period wise optimisation for a specific number of periods
- implement rolling optimisation? (might be too expensive)
- compute market_portfolio under a certain accuracy
- look into robustness ito trading days and choice of currency

* pf_diagnostic.py:
- implement diagnostics
    - Return, volatility, dividends
    - Sector, country, currency sensibility
    - Smallcap, largecap, liquidity
    - Firm leverage
- implement a function that checks impact of adding one stock to existing pf
- check correlation to an index or to a market
- overlap into optimisation
