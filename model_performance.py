import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import data_computer as dc


from sklearn.metrics import mean_squared_error, accuracy_score
#from sklearn.model_selection import cross_val_score, KFold
#%%
######################################################
#DEV MODEL
######################################################
"""
model = best_mdl
X_train = predictors_in
X_test = predictors_out_bt
y_train = target_in
y_test = target_out_bt
mlproblem = 'Regression'
"""
#%%
def get_predictions(model, X_train, y_train, X_test, y_test, plot=False, mlproblem="Regression"):
    """
    Return numpy array of predicted values based on input model and traning dataset.

    Parameters
    ----------
    model : Object.
        Scikit-learn already fitted machine learning model.
    X_test : pandas.DataFrame
        DataFrame with features data dedicated to make predictions.
    y_test : pandas.Series
        Series with real target values to evaluate model accuracy.
    mlproblem : string, optional
        "Regression" or "Classification". The default is "Regression".
        WARNING: be mindful and do not use continuous data for Classification problems.

    Returns
    -------
    predictions : numpy array
        Return predicted values in the same format as scikit-learn output.

    """
    predictions = model.predict(X_test)
    
    if mlproblem == "Classification": # for classification only
        print("accuracy:", accuracy_score(y_test, predictions)) 
    else:
        print('Mean squared error: %.2f' % (mean_squared_error(y_test, predictions)))
        print('R2 out-of sample multiplied by 100: %.2f' % (model.score(X_test, y_test)*100))

    # Scattered plot of the accuracy predictions
    if plot:    
        plt.figure(figsize=(15, 6))
        plt.scatter(model.predict(X_train), y_train, label='train')
        plt.scatter(model.predict(X_test), y_test, label='test')
        xmin, xmax = plt.xlim()
        plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')
        plt.legend()
        plt.show()

    return predictions

#example = get_predictions(model, X_test=X_test, y_test=y_test, mlproblem="Regression")
###############################################################################
#
# STOCKPICKING MODEL - SIMULATED PROC. END
#
###############################################################################
#%%
# Maximum drawdown
def max_drawdown(equity):
    """
    

    Parameters
    ----------
    equity : TYPE
        DESCRIPTION.
    startcash : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    if len(equity) > 0:
        maxdd = equity - (equity.cummax())
        peak = equity.cummax().iloc[-1]
        dt = maxdd.idxmin()
        return dt, maxdd.loc[dt]/peak, maxdd.loc[dt]
    else:
        return 0, None, None

#max_dd_stockpicker = max_drawdown(cumulative_equity)
#max_dd_hodl = max_drawdown(buy_n_hold_equity)
#%%
## TO DO: Cumulative returns of factor mimicking portfoli
# (reference: feature_selection.py)

# CAGR
def cagr(equity, start_cash):
    """
    

    Parameters
    ----------
    equity : TYPE
        DESCRIPTION.
    start_cash : TYPE
        DESCRIPTION.

    Returns
    -------
    cagr : TYPE
        DESCRIPTION.

    """
    first = start_cash
    last = equity[-1]
    annualization = (len(equity)/252)
    if first <= 0:
        print("Start capital must be larger than zero!")
    else:
        cagr = (last/first) ** (1/annualization) - 1 # scalar operations so I use plain operators
        return cagr

#cagr = cagr(cumulative_equity, start_cash=start_cash)
#%%
#%%
# Sharpe Ratio
def sharpe_ratio(equity, start_cash):
    
    """
    

    Parameters
    ----------
    equity : TYPE
        DESCRIPTION.
    start_cash : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if len(equity) > 1:
        #return (equity[-1] - start_cash) / (equity.diff().std()) # or equity.pct_change().mean() / equity.pct_change().std()?
        _, _, max_dd = max_drawdown(equity)
        sharpe_ratio = (equity[-1] - start_cash) / abs(max_dd)
        return sharpe_ratio
    else:
        return None

#cash = 100000
#sharpe_ratio_stockpicker = sharpe_ratio(cumulative_equity, start_cash=cash)
#sharpe_ratio_hodl = sharpe_ratio(buy_n_hold_equity, cash)
## Cumulative returns in the long-only portfolio
# CAGR
# Maximum drawdown
#%%
def change_frequency(df, freq_rebalanc):
    '''
    

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    freq_rebalanc : string
        DESCRIPTION.

    Returns
    -------
    df_resampled : TYPE
        DESCRIPTION.

    '''
    if freq_rebalanc=="monthly":
        df_resampled = df.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index") # first -> vytvori se NaNy
        idx_resampled = df_resampled.index.is_month_end
        idx_resampled[-1] = True # posledni neukonceny i ukonceny mesic bude zahrnut vzdy
        df_resampled = df_resampled.loc[idx_resampled] # jake hodnoty plati k tomu danemu konci obdobi (i kdyz neni business day)
    if freq_rebalanc=="quarterly":
        df_resampled = df.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index") # first -> vytvori se NaNy
        idx_resampled = df_resampled.index.is_quarter_end
        idx_resampled[-1] = True # posledni neukonceny i ukonceny kvartal bude zahrnut vzdy    
        df_resampled = df_resampled.loc[idx_resampled] 
    if freq_rebalanc=="annual":
        df_resampled = df.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index") # first -> vytvori se NaNy
        idx_resampled = df_resampled.index.is_year_end
        idx_resampled[-1] = True # posledni neukonceny i ukonceny rok bude zahrnut vzdy
        df_resampled = df_resampled.loc[idx_resampled] 
    
    return df_resampled
'''
df_resampled = change_frequency(df, freq_rebalanc)
'''
#%%
###############################################################################
#
# STOCKPICKING PROCEDURE 
#
###############################################################################


def backtest_stockpicker(model, 
                         df, 
                         X_train, 
                         y_train, 
                         X_test, 
                         y_test,
                         start_date,
                         freq_rebalanc,
                         desired_stocks,
                         leverage=1,
                         number_assets=3, 
                         start_cash=100000,
                         plot=False,
                         mlproblem="Regression"):
    '''
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    df : TYPE
        DESCRIPTION.
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    freq_rebalanc : TYPE
        DESCRIPTION.
    start_cash : TYPE, optional
        DESCRIPTION. The default is 100000.
    mlproblem : TYPE, optional
        DESCRIPTION. The default is "Regression".

    Returns
    -------
    stockpicker_pnl : TYPE
        DESCRIPTION.
    hodl : TYPE
        DESCRIPTION.

    '''
    #desired_stocks = list(set(X_test.index.levels[0]) - set(unwanted_stocks))
    X_train = X_train.loc[desired_stocks, start_date:, :].copy() # Remove unwanted Stocks and set start date
    y_train = y_train.loc[desired_stocks, start_date:, :].copy() # Remove unwanted Stocks and set start date
    X_test = X_test.loc[desired_stocks, start_date:, :].copy() # Remove unwanted Stocks and set start date
    y_test = y_test.loc[desired_stocks, start_date:, :].copy() # Remove unwanted Stocks and set start date
    
    # Resample Data
    if freq_rebalanc=="daily":
        df_resampled = df
    else:
        df_resampled = change_frequency(df=df, freq_rebalanc=freq_rebalanc)
        
    predictions = get_predictions(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, mlproblem=mlproblem) 
    predictions = pd.DataFrame(predictions, index=y_test.index, columns=['Predictions']).sort_index(axis="index") # redesign Array to DataFrame and sort index
    
    # Upsample to have daily dataframe
    predictions_daily_sample = predictions.unstack(level=0).resample(rule='1D').agg('last').bfill().stack(level=1)
    #y_test_daily_sample = y_test.unstack(level=0).resample(rule='1D').agg('last').bfill().stack()
    # Rebalance: matrix of stockpicking batches for a specific timestamp
    #rebalance_batch = predictions.unstack(level=0).swaplevel(axis="columns").droplevel(level=1, axis="columns").sort_index(axis="index").dropna()
    
    # Calculate 'daily' return (no cross-sectionally standardized)
    # df: Dataset with daily data
    return_no_scale = dc.compute_return(df) # RuntimeWarning: invalid value encountered in true_divide    #return_no_scale = y_test
    return_no_scale = return_no_scale.resample(rule='1D', axis='index').agg('first').fillna(method='ffill', axis='index') # fill even weekends and business holidays
    
    '''
    MIGHT BE REDUNDANT
    input_data = pd.concat(objs=[predictions_daily_sample, y_test_daily_sample], axis="columns")
    input_data.columns = ['Predictions', 'Return']
    # dropna replace with appropriate method (multivariate imputation by chained equations)
    input_data = input_data.unstack().swaplevel(axis="columns").sort_index(axis="index").sort_index(axis="columns").dropna()
    '''
    #timeSerie_test = predictions_daily_sample.sort_index(axis="index").index.unique() # plain time serie
    timeSerie_test = predictions_daily_sample.index.levels[0]
    stockpickerLong_pnl = pd.Series(dtype="float64")
    stockpickerShort_pnl = pd.Series(dtype="float64")
    
    #TO DO: VECTORIZE
    for idx in timeSerie_test:
        # level=1 => Stock symbols in index, Prediction and Return in columns
        rebalance = predictions_daily_sample.unstack().droplevel(level=0, axis='columns').loc[idx].sort_values(ascending=False) # batch for 1 timestamp only
        # Long Leg
        long = rebalance.iloc[:number_assets]
        stockpickerLong_pnl = stockpickerLong_pnl.append(pd.Series(return_no_scale.loc[idx, long.index].mean() * leverage)) # idx is a specific day, long.index is for appropriate stocks
        # Short Leg
        short = rebalance.iloc[-number_assets:]
        stockpickerShort_pnl = (stockpickerShort_pnl.append(pd.Series(return_no_scale.loc[idx, short.index].mean()*-1))) # -1 is for Short Direction
        
        print("*******************************************")
        print(idx)
        print("Picked Stocks - LONG: ", list(long.index))
        print("Picked Stocks - SHORT: ", list(short.index))
        print("Long Result: ", stockpickerLong_pnl.iloc[-1])
        print("Short Result: ", stockpickerShort_pnl.iloc[-1])
        print("Overall Result: ", np.mean([stockpickerLong_pnl.iloc[-1], stockpickerShort_pnl.iloc[-1]]))
        print("*******************************************")

    # Dataset with Results
    stockpicker_pnl = pd.concat(objs=[stockpickerLong_pnl, stockpickerShort_pnl], axis="columns")
    stockpicker_pnl.columns = ["Long Only", "Short Only"]
    stockpicker_pnl.index = timeSerie_test
    
    stockpicker_pnl["Stockpicker_Return"] = stockpicker_pnl.mean(axis="columns")
    stockpicker_pnl["Portfolio_Cumulative_Return"] = stockpicker_pnl["Stockpicker_Return"].cumsum()
    
    ########################
    # BUY AND HOLD RETURNS
    ########################
    # Buy and Hold Returns for All Stocks
    hodl = dc.compute_return(df).loc[timeSerie_test[0]:, :].copy()
   
    hodl["Buy_N_Hold_Return"] = hodl.mean(axis="columns")
    hodl["Buy_N_Hold_Cumulative_Return"] = hodl["Buy_N_Hold_Return"].cumsum()
    ########################
    # BUY AND HOLD END
    ########################
    
    ########################
    # MONETIZE BACKTEST
    ########################
    stockpicker_pnl["Portfolio_Cumulative_Equity"] = start_cash + stockpicker_pnl['Portfolio_Cumulative_Return'] * start_cash
    hodl["Buy_N_Hold_Cumulative_Equity"] = start_cash + hodl['Buy_N_Hold_Cumulative_Return'] * start_cash
    
    print("*******************************************")
    print("Stockpicker Equity: ", stockpicker_pnl["Portfolio_Cumulative_Equity"].iloc[-1], "USD")
    print("Buy N Hold Equity: ", hodl["Buy_N_Hold_Cumulative_Equity"].iloc[-1], "USD")
    print("*******************************************")

    # Plot the Whole Portfolio Cumulative Return  
    if plot:
        stockpicker_pnl["Portfolio_Cumulative_Equity"].plot(label='Portfolio Cumulative Return')
        plt.title("Stockpicker Cumulative Equity")
        plt.ylabel("Equity")
        plt.show()
        # Plot the Whole Buy and Hold Cumulative Return
        hodl["Buy_N_Hold_Cumulative_Equity"].plot(label="Buy N Hold Cumulative Return")
        plt.title("Buy N Hold Cumulative Equity")
        plt.ylabel("Equity")
        plt.show()
    
    """
    # Check Trades where was portfolio return different form buy and hold 
    mask_trades = np.where(hodl["Buy_N_Hold_Return"] != portfolio["Portfolio_Return"], True, False)
    different_trades = portfolio.loc[mask_trades, "Portfolio_Return"]
    
    mask_gain_trades = np.where(different_trades>0, True, False)
    gain_trades = different_trades.loc[mask_gain_trades]
    """
    ########################
    # MONETIZE BACKTEST END
    ########################
       
    ########################
    # PERFORMANCE 
    ########################
    cagr_sp = cagr(stockpicker_pnl["Portfolio_Cumulative_Equity"], start_cash=start_cash)
    cagr_hodl = cagr(hodl["Buy_N_Hold_Cumulative_Equity"], start_cash=start_cash)
    print("*******************************************")
    print("Stockpicker CAGR: ", cagr_sp)
    print("Buy N Hold CAGR: ", cagr_hodl)
    print("*******************************************")
    
    # Maximum Drawdown
    dd_sp = max_drawdown(equity=stockpicker_pnl["Portfolio_Cumulative_Equity"])
    dd_hodl = max_drawdown(equity=hodl["Buy_N_Hold_Cumulative_Equity"])
    print("*******************************************")
    print("Stockpicker MAX Drawdown: ", dd_sp)
    print("Buy N Hold MAX Drawdown: ", dd_hodl)
    print("*******************************************")
    
    # Sharpe Ratio
    sr_sp = sharpe_ratio(equity=stockpicker_pnl["Portfolio_Cumulative_Equity"], start_cash=start_cash)
    sr_hodl = sharpe_ratio(equity=hodl["Buy_N_Hold_Cumulative_Equity"], start_cash=start_cash)
    print("*******************************************")
    print("Stockpicker Sharpe Ratio: ", sr_sp)
    print("Buy N Hold Sharpe Ratio: ", sr_hodl)
    print("*******************************************")
    
    return stockpicker_pnl, hodl

#stockpicker_pnl, hodl = backtest_stockpicker(model=model, X_test=X_test, y_test=y_test)
#%%
# Next Rebalance Batch
def next_rebalance(model, X_test, plot=False, generate_csv='False'):
    '''
    Function returns next rebalance batch.

    Parameters
    ----------
    model : scikit-learn object.
        Fitted model, scikit-learn object..
    X_test : pd.DataFrame
        Test predictors data.
    generate_csv : boolean, optional
        True value generates csv report. The default is 'False'.

    Returns
    -------
    target_fcast_decomposed : pandas.DataFrame
        Output report.

    '''
    
    model_const = model.intercept_
    model_betas = pd.Series(data=model.coef_, index=X_test.columns)
    predictors_last = X_test.loc[:, X_test.index[-1], :] # bla4 ve StockPicking_ComputeForecasts.m
    rebalance_date = str(X_test.index.droplevel(0)[-1])[:10]
    predictors_last = predictors_last.droplevel(level=1, axis='index')
    
    target_fcast_decomposed = predictors_last.multiply(other=model_betas, axis="columns")
    target_fcast_decomposed["Constant"] = model_const
    target_fcast_decomposed['Score'] = target_fcast_decomposed.sum(axis='columns')
    
    # Score a konstantu na zacatek
    target_fcast_decomposed = pd.concat(objs=[target_fcast_decomposed['Score'], target_fcast_decomposed["Constant"], target_fcast_decomposed.drop(labels=["Score", "Constant"], axis="columns")],
                                        axis="columns") # DECOMPOSED SCORE - bla6 ve StockPicking_ComputeForecasts.m (+ konstanta)
    #target_fcast_decomposed_fin = target_fcast_decomposed.sum(axis="columns") # jen pro kontrolu s target_fcast_final
    target_fcast_decomposed = target_fcast_decomposed.sort_values(by='Score', ascending=False)
    
    # Plot Score Heatmap
    if plot:
        labels = list(target_fcast_decomposed.index) # y-axis captions
        sns.heatmap(target_fcast_decomposed.drop(['Score', 'Constant'], axis='columns'), annot=True, linewidths=.5, yticklabels=labels)
        
    if generate_csv:
        target_fcast_decomposed.to_csv('rebalance_' + rebalance_date + '.csv')
    
    return target_fcast_decomposed

'''
rebalance_batch = next_rebalance(model=best_mdl, X_test=predictors_out, generate_csv=True)
'''