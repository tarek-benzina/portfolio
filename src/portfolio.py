import numpy as np
import pandas as pd
from scipy.optimize import minimize

def portfolio_buy_date_return(tickers, shares, stock_log_return, buy_date, price_column, log_return_column="log_return"):
    # Filter stock_log_return based on buy_date
    tmp = stock_log_return[stock_log_return['date'] == buy_date].copy()
    # Create a DataFrame for ticker shares
    ticker_shares = pd.DataFrame({'ticker': tickers, 'share': shares})
    # Merge ticker_shares with filtered stock_log_return
    tmp = tmp.merge(ticker_shares, on='ticker', how='left')
    # Calculate value and budget for each date
    tmp['value'] = tmp['share'] * tmp[price_column]
    tmp['budget'] = tmp.groupby('date')['value'].transform('sum')
    # Calculate the weight for each row
    tmp['weight'] = tmp['value'] / tmp['budget']
    # Calculate the portfolio log return
    portfolio_log_return = np.log(np.sum(tmp['weight'] * np.exp(tmp[log_return_column])))
    return portfolio_log_return

def portfolio_weights(tickers,shares,stock_log_return,buy_date,price_column):
    tmp=stock_log_return.copy()
    tmp=tmp[(tmp.date==buy_date)]
    ticker_shares=pd.DataFrame(tickers,columns=["ticker"])
    ticker_shares["share"]=shares
    tmp=tmp.merge(ticker_shares,on="ticker",how="left")
    tmp["value"]=tmp["share"]*tmp[price_column]
    tmp["weight"]=tmp["value"]/tmp["value"].sum()
    return tmp[["ticker","share","value","weight"]]

def portfolio_diversity(tickers,shares,stock_log_return,buy_date,price_column):
    res=portfolio_weights(tickers,shares,stock_log_return,buy_date,price_column)
    return sum(res["weight"]**2)
    
def investment(tickers,shares,stock_log_return,buy_date,price_column):
    tmp=stock_log_return.copy()
    tmp=tmp[(tmp.date==buy_date)]
    ticker_shares=pd.DataFrame(tickers,columns=["ticker"])
    ticker_shares["share"]=shares
    tmp=tmp.merge(ticker_shares,on="ticker",how="left")
    tmp["value"]=tmp["share"]*tmp[price_column]
    return tmp["value"].sum()

def compute_risk(data,tickers,shares,log_return_column="log_return"):
    weights=shares/shares.sum()## fix weight calc
    returns=data.pivot(index="date",columns=["ticker"],values=log_return_column)
    return np.sqrt(returns[tickers].cov().mul(weights,axis=0).mul(weights,axis=1).sum().sum())

#def value_at_risk(data,tickers,shares)
from scipy.stats import norm
def compute_value_at_risk(data, tickers, shares,buy_date,price_column="Close", log_return_column="log_return", confidence_level=0.95):
    # Compute portfolio variance
    portfolio_risk = compute_risk(data[data.date<buy_date], tickers=tickers, shares=shares, log_return_column=log_return_column)
    portfolio_value = investment(tickers=tickers,shares=shares,stock_log_return=data,buy_date=buy_date,price_column=price_column)
    
    # Calculate the z-score for the desired confidence level
    z = norm.ppf(1 - confidence_level)
    
    # Calculate VaR
    var = -portfolio_value*z * portfolio_risk
    
    return var
def optimise(prediction_results,date,max_budget,min_budget,tickers,max_value_at_risk,predicted_log_return_column,log_return_column,expected_price_column,min_log_return=-1,diversity=.1):
    if min_budget>max_budget:
        raise ValueError("Min budget cannot be higher than max budget")

    drop_tickers=prediction_results[
        (prediction_results.date==date)
        &(prediction_results.ticker.isin(tickers))
        &((prediction_results[predicted_log_return_column].isnull())|(prediction_results[predicted_log_return_column]>min_log_return))
    ].ticker.unique()
    
    print(f"Dropping the following ticker",drop_tickers)
    tickers=[t for t in tickers if t not in drop_tickers]
    if len(tickers)>10:
        if len(prediction_results)==0:
            raise ValueError("no data to optimise")
        cons = ({'type': 'ineq', 'fun': lambda x: max_value_at_risk-compute_value_at_risk(data=prediction_results,
                                                                                   tickers=tickers, 
                                                                                   shares=x,
                                                                                   buy_date=date,
                                                                                   price_column=expected_price_column, 
                                                                                   log_return_column=log_return_column,
                                                                                   confidence_level=0.99)},
                {'type': 'ineq', 'fun': lambda x: diversity-portfolio_diversity(tickers=tickers, 
                                                                                   shares=x,
                                                                                stock_log_return=prediction_results,
                                                                                   buy_date=date,
                                                                                   price_column=expected_price_column)},
           {'type': 'ineq', 'fun': lambda x: max_budget-investment(tickers,x,prediction_results,buy_date=date,price_column=expected_price_column)},
           {'type': 'ineq', 'fun': lambda x: -min_budget+investment(tickers,x,prediction_results,buy_date=date,price_column=expected_price_column)}
           )
    
        bnds = [(0, None) for _ in tickers]
        init_guess = [1e-5] * len(tickers)
    
        func=lambda x: - portfolio_buy_date_return(
            tickers=tickers,
            shares=x,
            stock_log_return=prediction_results,
            buy_date=date,
            price_column=expected_price_column,
            log_return_column=predicted_log_return_column)
    
        res = minimize(func, init_guess, method='COBYLA', bounds=bnds,
                       constraints=cons)
        return  res,res.x,tickers
    else: 
        return np.zeros(len(tickers)),tickers

def get_historical_returns(data,date,log_return_col):
    hist=data[data.date<date][["ticker","date",log_return_col]].dropna()
    hist=hist.pivot(index="date",columns=["ticker"],values=log_return_col).fillna(0).to_numpy()
    return hist
hhi=lambda shares,prices: np.sum(prices*shares/(np.sum(prices*shares))**2)

portfolio_expected_investment=lambda shares,prices: np.sum(shares*prices)
portfolio_expected_return=lambda shares,prices,log_returns: np.log(np.sum(prices*shares/(np.sum(prices*shares))*np.exp(log_returns)))
risk = lambda hist,shares,prices: np.sqrt(np.cov(hist.transpose()).dot(prices*shares/(np.sum(prices*shares))).dot(prices*shares/(np.sum(prices*shares))))
var = lambda hist,shares,prices,z: -portfolio_expected_investment(shares,prices=prices)*z * risk(hist,shares,prices)

def maximise_returns(result,
                     date,
                     expected_price_col,
                     expected_log_return_col,
                     log_return_col,
                     log_return_horizon,
                     min_expected_log_return=0.,
                     max_value_at_risk=10,
                     confidence_level=.95,
                     max_budget=1000,
                     min_budget=50,
                     diversity=.4,
                    ):
    ticker_list=result[(result.date==date)
    &(result[expected_log_return_col]>=min_expected_log_return)
    &(~result[expected_price_col].isnull())
    &(~result[expected_log_return_col].isnull())
    ].ticker.unique()
    print(ticker_list)
    if len(ticker_list)>10:
        result=result[result.ticker.isin(ticker_list)]

        z = norm.ppf(1 - confidence_level)
        res=result[result.date==date][["ticker",expected_price_col,expected_log_return_col]]
        hist=result[result.date<date-pd.Timedelta(days=log_return_horizon)][["ticker","date",log_return_col]]
        hist=get_historical_returns(data=result,date=date,log_return_col=log_return_col)
        prices=res[expected_price_col].to_numpy()
        log_returns=res[expected_log_return_col].to_numpy()



        cons = ({'type': 'ineq', 'fun': lambda x: max_value_at_risk-var(hist=hist,shares=x,prices=prices,z=z)},
                {'type': 'ineq', 'fun': lambda x: diversity-hhi(shares=x,prices=prices)}, 
                #                                                                    shares=x,
                #                                                                stock_log_return=prediction_results,
                #                                                                    buy_date=date,
                #                                                                    price_column=expected_price_column)},
            {'type': 'ineq', 'fun': lambda x: max_budget-portfolio_expected_investment(shares=x,prices=prices)},
            {'type': 'ineq', 'fun': lambda x: -min_budget+portfolio_expected_investment(shares=x,prices=prices)}
            )
        bnds = [(1e-10, None) for _ in ticker_list]
        init_guess = [1] * len(ticker_list)
        func=lambda x : -portfolio_expected_return(shares=x,prices=prices,log_returns=log_returns)
        res = minimize(func, init_guess, method='SLSQP', bounds=bnds,
                    constraints=cons,#options={"max_iter":100}
                    )
        print(res)
        return ticker_list,res.x,res.success
    else:
        return ticker_list,np.zeros(len(ticker_list)),False
