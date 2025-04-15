TODO:

* optimisation.py: 
- add return output showing actual start and actual end
- implement period wise optimisation for a specific number of periods
- implement rolling optimisation? (might be too expensive)
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

* estimation.py:
- create a multiples estimation tool
- create a DCF estimation tool. params:
    - return_on_equity= (float64, "risk", "capm", "peer")
    - cost_of_debt= (float64, "interest", "credit_spread", "synthetic_rating")
    - apply_size_premium= (true, false)
    - apply_liquidity_premium= (true, false)
    - apply_country_risk_premium= (true, false)
    - revenue_increase (final value will be considered perpetual growth rate)
    - cost_of_revenue_increase 
    - operating_expenses_increase