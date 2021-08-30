import pandas as pd
import numpy as np
import talib as ta

from pyfinance import ols #https://stackoverflow.com/questions/64497720/modulenotfounderror-no-module-named-pandas-libs-tslibs-frequencies
from statsmodels.tsa.stattools import acovf
import statsmodels as sm
import os

from sklearn.preprocessing import StandardScaler

"""
# funkce pro pocitani (a souvisejici s pocitanim) faktoru 
# .. pocitaji se vzdy jen daily (nejpodrobnejsi) - dle dalsich potreb (frekv. rebalance atd.) se pak prevzorkuje
# .. nektere fce pro vypocet faktoru navazne pouziva data_storage (ukladani, dopocet atd.) - zmena zde v dc muze zpusobit problem v ds
"""

#%%
def add_feature(basic_df:pd.DataFrame, feature:pd.DataFrame, level=0, feature_name="feature"):
    """
    Add Feature Function:
    -------------------
    Function create new columns for computed feature (dataframe) and add them into a basic dataframe under an appropriate column (Symbol).
    # Create List of Column Names (in specific Level).
    # Redesign index and columns structure for a new feature
    # Concatenate a new feature DataFrame and the basic dataset
    Assumption: Level1 is a list of Symbols and Level2 are Features.
    
    Parameters
    ----------
    basic_df
        Dataframe with other features where the new one should be added.
    feature: DataFrame
        A variable of computed feature.
    level: integer
        A level where the feature should be added.
    feature_name: str
        Name of the new feature.Valid for Level 1 Columns. For Level 0 is the feature name taken from input DataFrame.
    """
    if level == 0:
        # Final Concatenation of new Feature into Basic Dataset
        df = pd.concat(objs=[basic_df, feature], axis=1, levels=1)
    
    if level == 1:
        #new_cols_l0 = list(basic_df.columns.levels[0]) # Create List of Column Names in Level 0
        new_cols_l0 = basic_df.columns.levels[0]
        new_cols_l1 = [feature_name for item in new_cols_l0] # Create List of Column Names in Level 1
        col_names = tuple(zip(new_cols_l0, new_cols_l1)) # Complete Multiindex
        
        feature = pd.DataFrame(feature.to_numpy(), index=basic_df.index, columns=col_names) # Redesign Index and Column Names
        # Final Concatenation of new Feature into Basic Dataset
        df = pd.concat(objs=[basic_df, feature], axis=1).sort_index(axis=1)
    
    return df
    
#%%

def compute_return(df):
    '''
    Function calculates Firm-Specific Returns and add Columns into Level 1.

    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame where is input data and where is the new feature added.

    Example
    -------
    >>> df = add_return(df)
    >>> df
    Symbol    AAPL    
    Features    Close    Dividends    High    Low    Open    Return    Stock Splits    Volume ...
    Date                                                                                
    2021-04-15    134.500000    0.0    135.000000    133.639999    133.820007    0.018535    0.0    89260600 ...
    
    '''
    # Calculate Firm-Specific Return
    df_return = pd.DataFrame(np.log(df.loc[:, (slice(None), "Close")].to_numpy()/df.loc[:, (slice(None), "Close")].shift(1).to_numpy()), index=df.index, columns=df.columns.levels[0]) # jen hodnoty "bez" columns, resp. 1-levelove columns
    #scaled_return = standardize(df_return)
    # Add new feature to the dataset
    #feature = add_feature(df, scaled_return, level=1, feature_name="Return")
    feature = df_return
    
    return feature

#%%
#data = read_yf_data(tickers=["AAPL", "FB", "AMZN", "MSFT", "IBM", "F", "CVX"])
#df = compute_return(data)

def cross_sectional_standardize(to_scaling):
    """
    Perform scaling (standardization) based on scikit-learn StandardScaler:
    Standard Score (output: u=0, sigma=1)
    z = (input-mean)/sqrt(var)

    Parameters
    ----------
    to_scaling : pandas.DataFrame
        Input tabular data dedicated to be scaled.
            a) Both hiearchical and tabular data (level 0: feature, level 1: Symbol)
            b) Input data has to be shaped "time X assets"
    Returns
    -------
    scaled : pandas.DataFrame
        Output scaled data.
    
    Example
    -------
    >>> df_currvol
    Out[282]: 
                           0             1  ...             5             6
    Date                                    ...                            
    2010-06-01           NaN           NaN  ...           NaN           NaN
    2021-06-18  1.446541e+11  1.513876e+11  ...  8.881246e+09  8.841680e+10
    2021-06-21  1.462960e+11  1.551847e+11  ...  9.145392e+09  9.086157e+10
    
    [2784 rows x 7 columns]
    
    >>> standardize(df_currvol)
    Out[283]: 
                       0         1         2  ...         4         5         6
    Date                                      ...                              
    2010-06-01       NaN       NaN       NaN  ...       NaN       NaN       NaN
    2021-06-18  1.308983  1.430788 -1.004363  ...  0.030408 -1.147051  0.291691
    2021-06-21  1.291203  1.448527 -0.989241  ...  0.007753 -1.136274  0.310050
    
    [2784 rows x 7 columns]
    """        
    scaler = StandardScaler()
    #to_scaling = to_scaling.droplevel(level=1, axis="columns").T
    transposed = to_scaling.T # transpose matrix to perform cross-sectional standardization
    scaled = pd.DataFrame(scaler.fit_transform(transposed), index=transposed.index, columns=transposed.columns).T
    
    return scaled
#%%
def make_standardization(features_to_scale):
    '''
    Standardize cross-sectionally all input features:
        a) slice specific feature for all stocks
        b) standardize values among all stocks
    
    Parameters
    ----------
    features_to_scale : pandas.DataFrame
        Input features determined to sta.

    Returns
    -------
    cross_sectionally_scaled : TYPE
        DESCRIPTION.

    '''
    cross_sectionally_scaled = pd.DataFrame()
    
    for feature, feature_df in features_to_scale.groupby(axis=1, level=1):
        #print(feature_df)
        cross_sectionally_scaled_tmp = cross_sectional_standardize(to_scaling=feature_df)
        
        cross_sectionally_scaled = pd.concat(objs=[cross_sectionally_scaled, cross_sectionally_scaled_tmp], axis="columns")
    
    return cross_sectionally_scaled

'''
cross_sectionally_scaled = make_standardization(features_to_scale)
'''
#%%
# Calculate Benchmark Return (common for all stocks)
# Selection second Level (Firm-Specific) Returns

def benchmark_return(df):
    '''
    Function returns Benchmark Return (Cross-Sectional) and Cumulative Benchmark Return added in the inputed DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame
      Input DataFrame where is input data and where is the new feature added.
      
    Example
    -------
    >>>feature = benchmark_return(df)
    >>>feature
    Symbol	AAPL	AMZN	...	MSFT	Benchmark_Return Cumulative_Benchmark_Return
    Features	Close	Dividends	High ...
    2021-04-15	134.500000	...	0.007895 1.825752
    '''
    # Select Firm-Specific Return
    mean_return = df.loc[:, (slice(None), "Return")].copy() # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
    
    # Calculate cross-sectional (axis=1) Benchmark Return 
    mean_return["Benchmark_Return"] = mean_return.mean(axis=1)
    mean_return["Cumulative_Benchmark_Return"] = mean_return["Benchmark_Return"].cumsum()
    feature = mean_return[["Cumulative_Benchmark_Return", "Benchmark_Return"]]
          
    return feature

#feature = benchmark_return(features_all)

#%%

def currency_volume(df: pd.DataFrame, horizons: list, horizon_pairs: list):
    """
    Computes rolling currency volume, rolling currency volume absolute and relative change

    Parameters
    ----------
    df : pandas.DataFrame
        input DataFrame with "raw" prices etc. data
    horizons : list
        lenght of horizons for rolling calculation
    horizon_pairs : list of lists
        pairs of short and long horizons, between which the change is calculated
    """
    # get unique horizons from all horizons
    horizons_unq = horizons.copy()
    horizons_unq.extend(np.unique(horizon_pairs)) 
    horizons_unq = np.unique(horizons_unq)
    
    # Calculate Firm-Specific Dollar Volume
    df_close = df.loc[:, (slice(None), "Close")]
    df_volume = df.loc[:, (slice(None), "Volume")]

    df_currvol_tmp = pd.DataFrame(df_volume.to_numpy()*df_close.to_numpy(), index=df.index) # HM_USDVol - taky vystup
    df_currvol_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["CurrencyVolume_1"] ])
    # zachovat index - kvuli diferencovani, kde by se to zmizenim 1. polozky posunulo
    #df_dcurrvol_tmp = df_currvol_tmp.diff(periods=1, axis="index") # HM_DUSDVol - Ficura dal nepouziva
    
    features = df_currvol_tmp.copy() #pd.DataFrame() # jen pozadovane features    
    features_tmp = pd.DataFrame() 
    
    # firm-specific currency volume
    for h in horizons_unq:
        df_currvol = df_currvol_tmp.rolling(window=h, axis="index").sum()
        df_currvol.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["CurrencyVolume_{}".format(h)] ])
        #scaled_currvol = standardize(df_currvol) # scaling data/data normalization
        # vsechny features potrebne pro dalsi vypocet
        features_tmp = pd.concat(objs=[features_tmp, df_currvol],  axis="columns")
        
        # jen features pozadovane na vystupu
        if h in horizons:
            features = pd.concat(objs=[features, df_currvol],  axis="columns")
            #features = pd.concat(objs=[features, scaled_currvol],  axis="columns")

    # firm-specific currency volume absolute and relative change 
    for h_short, h_long in horizon_pairs:
        df_avgshort = features_tmp.loc[:, (slice(None), "CurrencyVolume_{}".format(h_short))]/h_short
        df_avglong = features_tmp.loc[:, (slice(None), "CurrencyVolume_{}".format(h_long))]/h_long
        df_chng_abs = df_avgshort.to_numpy() - df_avglong.to_numpy()
        captions = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["CurrencyVolumeAbsChange_{}vs{}".format(h_short, h_long)] ])
        df_chng_abs = pd.DataFrame(data=df_chng_abs, index=df.index, columns=captions)
        #scaled_chng_abs = standardize(df_chng_abs)        
        
        df_chng_rel = (df_avgshort.to_numpy()/df_avglong.to_numpy())-1
        captions = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["CurrencyVolumeRelChange_{}vs{}".format(h_short, h_long)] ])
        df_chng_rel = pd.DataFrame(data=df_chng_rel, index=df.index, columns=captions)
        #scaled_chng_rel = standardize(df_chng_rel)
        
        features = pd.concat(objs=[features, df_chng_abs, df_chng_rel], axis="columns")
        #features = pd.concat(objs=[features, scaled_chng_abs, scaled_chng_rel], axis="columns")

            
    return features

#features_new = currency_volume(data, horizons, horiz_pairs)

#%%

def compute_volume(df):
    '''
    Function calculates Cumulative Volume (used for weekly and monthly aggregation).
    
    Parameters
    ----------
    df: pandas.DataFrame
      Input DataFrame where is input data and where is the new feature added.
      
    Example
    -------
    >>>df_new = compute_volume(df)
    ...Symbol            AAPL                     ...         MSFT          
    Features         Close Cummulative_Volume  ... Stock Splits    Volume
    Date                                       ...                       
    2021-04-16  134.259903       7.327248e+11  ...            0   7741157
    '''
    # Calculate Cumulative Volume and add it into the input Dataset
    feature = df.loc[:, (slice(None), "Volume")].cumsum()
    #scaled_feature = standardize(feature)
        
    return feature

#df_new = compute_volume(data)

#%%
def statistical_moments(df:pd.DataFrame, horizons:list):
    '''
    Function calculates Statistical Moments: mean, st. deviation, variance, skewness, kurtosis, autoocorrelation.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame where is input data and where is the new feature added.
    horizons: list
        List of integers as window lenghts.
    horizon_pairs: list of lists
        Pairs of short and long horizons, between which the change is calculated
    
    Example
    -------
    >>>df_stat = statistical_moments(df, horizons)    
    >>>print(df_stat)
    ...
    Symbol            AAPL            ...         MSFT          
    Features         Close Dividends  ... Volatility_5    Volume
    Date                              ...                       
    2010-06-01    7.980511       0.0  ...          NaN  76152400
    2010-06-02    8.075969       0.0  ...          NaN  65718800
    2021-04-15  134.500000       0.0  ...     0.010535  25627500
    2021-04-16  134.160004       0.0  ...     0.010118  24856900
    '''
    
    returns = compute_return(df) # only returns for selected stocks
    returns.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Return"]])
    df_tmp = returns
    
    features = pd.DataFrame()
    
    for h in horizons:
        mean = df_tmp.rolling(window=h).mean()
        mean.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Mean_{}".format(h)]])
        #scaled_mean = standardize(mean)
        
        volatility = df_tmp.rolling(window=h).std()
        volatility.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volatility_{}".format(h)]])      
        #scaled_volatility = standardize(volatility)
        
        variance = df_tmp.rolling(window=h).var()
        variance.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Variance_{}".format(h)]])      
        #scaled_variance = standardize(variance)
        
        skewness = df_tmp.rolling(window=h).skew()
        skewness.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Skewness_{}".format(h)]])     
        #scaled_skewness = standardize(skewness)
        
        kurtosis = df_tmp.rolling(window=h).kurt()
        kurtosis.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Kurtosis_{}".format(h)]])
        #scaled_kurtosis = standardize(kurtosis)
        
        t_statistic = pd.DataFrame(np.divide(mean.to_numpy(), volatility.to_numpy() * np.sqrt(h)), index=df.index)
        t_statistic.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["T_statistic_{}".format(h)]])
        #scaled_t_statistic = standardize(t_statistic)
        

        # Pretty slow but...
        # https://stackoverflow.com/questions/51453152/computing-rolling-autocorrelation-using-pandas-rolling
        # https://stackoverflow.com/questions/36038927/whats-the-difference-between-pandas-acf-and-statsmodel-acf
        autocorrelation = df_tmp.rolling(window=h).apply(pd.Series.autocorr)
        autocorrelation.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Autocorrelation_{}".format(h)]])
        #scaled_autocorrelation = standardize(autocorrelation)

        #autocov = df_tmp.rolling(window=h).apply(acovf) # demean=True je default, DataError: No numeric types to aggregate
        #autocov = df_tmp.rolling(window=h).apply(lambda x_win: acovf(x=x_win, demean=True)) # DataError: No numeric types to aggregate
        autocov = pd.DataFrame(np.multiply(autocorrelation.to_numpy(), variance.to_numpy() ), index=df.index) # LZE TO TAKTO? acorr = acov / var
        autocov.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Autocovariance_{}".format(h)]])
        
        median = df_tmp.rolling(window=h).median()
        median.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Median_{}".format(h)]])
        #scaled_median = standardize(median)           
        
        mean_minus_med = pd.DataFrame(np.subtract(mean.to_numpy(), median.to_numpy() ), index=df.index)
        mean_minus_med.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Mean_minus_med_{}".format(h)]])

        # Penultimate concatenation to the 'feature' dataframe    
        features = pd.concat(objs=[features,
                                   mean, 
                                   volatility, 
                                   variance, 
                                   skewness, 
                                   kurtosis, 
                                   t_statistic, 
                                   autocov, # autocovar chybela (Ficura ji spocita, ale nedava do vystupu) = acorr * var (?) - bude bez standardizace
                                   autocorrelation, 
                                   median,
                                   mean_minus_med # chybelo - bude bez standardizace atd.
                                   ], axis=1).sort_index(axis=1)
                       
    return features

#df_statistical_moments = statistical_moments(features_all, horizons)    
#%%
# Momentum Differences
def momentum(df, horizons, horizon_pairs):
    '''
    Function calculates momentum for specific horizons and momentum differencies.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame where is input data and where is the new feature added.
    horizons: list
        List of integers as window lenghts.
    horizons_pairs: list of lists
        must contain numbers only from horizons list
    
    Example
    -------
    >>>horizons = [5, 10, 22]
    >>>horizon_pairs = [[5, 10], [5, 22], [10, 22]]
    >>>df_momentum = momentum(df, horizons, horizon_pairs)
    ...
                AAPL                ...          MSFT           
                Momentum_10 Momentum_10-5  ... Momentum_22-5 Momentum_5
    Date
    021-04-16    0.086849      0.078165  ...      0.073540   0.018932
    2021-04-19    0.068601      0.041540  ...      0.076597   0.010998
    2021-04-20    0.053229      0.063096  ...      0.113652  -0.000890
    2021-04-21    0.042853      0.031780  ...      0.103975   0.019335
    2021-04-22    0.012047      0.031264  ...      0.094967  -0.009019
    '''
    returns = compute_return(df) # only returns for selected stocks
    returns.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Return"]])
    df_tmp = returns
    
    features = pd.DataFrame()
    
    for h in horizons:
        df_momentum = df_tmp.rolling(window=h).sum()
        df_momentum.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Momentum_{}".format(h)]])
        #scaled_momentum = standardize(df_momentum)
        
        features = pd.concat(objs=[features, df_momentum], axis=1).sort_index(axis=1)
        
    for h_short, h_long in horizon_pairs:
        df_momentum_long = pd.DataFrame()
        df_momentum_short = pd.DataFrame()
         
        df_momentum_long = features.loc[:, (slice(None), "Momentum_{}".format(h_long))]
        df_momentum_short = features.loc[:, (slice(None), "Momentum_{}".format(h_short))]
        
        df_momentum_diff = df_momentum_long.to_numpy() - df_momentum_short.to_numpy()
        captions = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Momentum_{}-{}".format(h_long, h_short)]])
        df_momentum_diff = pd.DataFrame(data=df_momentum_diff, index=df.index, columns=captions)
        #scaled_momentum_diff = standardize(df_momentum_diff)
        
        features = pd.concat(objs=[features, df_momentum_diff], axis=1).sort_index(axis=1)
        
    return features

#df_momentum = momentum(features_all, horizons, horizon_pairs)

#%%

def technical_factors(df: pd.DataFrame, horizons: list):
    '''
    Function calculates price technical indicators for specific horizons.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame where is input data and where is the new feature added.
    horizons: list
        List of the investment horizons.
    Example
    -------
    >>>technical_features = technical_factors(df)
    ...
                           AAPL                         ...       MSFT                       
                ARRONOSC_10 ARRONOSC_22 ARRONOSC_5  ...   WILLR_22    WILLR_5      WMA_22
    Date                                           ...                                  
                      ...         ...        ...  ...        ...        ...         ...
    2021-04-26        60.0   63.636364      -40.0  ...  -2.839868 -13.088445  255.919010
    2021-04-27       -20.0   63.636364      -40.0  ...  -3.801812 -16.158950  256.890156
    2021-04-28       -20.0   63.636364       60.0  ... -26.893131 -84.277346  257.099840
    2021-04-29        50.0   95.454545      100.0  ... -33.281425 -75.264314  257.059800
    2021-04-30       -10.0   95.454545      -20.0  ... -35.746781 -77.589907  256.922803
    '''
    # Technical Factors
    df_close = df.loc[:, (slice(None), "Close")]
    
    df_close = df_close.apply(pd.to_numeric, errors="coerce")
    
    # All features declaration
    rsi_df = pd.DataFrame()
    ema_df = pd.DataFrame()
    kama_df = pd.DataFrame()
    wma_df = pd.DataFrame()
    cmo_df = pd.DataFrame()
    mom_df = pd.DataFrame()
    roc_df = pd.DataFrame()
    rocp_df = pd.DataFrame()
    rocr100_df = pd.DataFrame()
    ht_trendline_df = pd.DataFrame()
    aroonosc_df = pd.DataFrame()
    cci_df = pd.DataFrame()
    dx_df = pd.DataFrame()
    #mfi_df = pd.DataFrame()
    minus_di_df = pd.DataFrame()
    plus_di_df = pd.DataFrame()
    plus_dm_df = pd.DataFrame()
    willr_df = pd.DataFrame()
    atr_df = pd.DataFrame()
    natr_df = pd.DataFrame()

    for h in horizons:
        # RSI
        rsi_tmp = df_close.apply(lambda close: ta.RSI(close, h))
        rsi_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["RSI_{}".format(h)]])
        rsi_df = pd.concat(objs=[rsi_df, rsi_tmp], axis=1).sort_index(axis=1)
        #scaled_rsi = standardize(rsi_df)
        
        # EMA - Exponential Moving Average
        ema_tmp = df_close.apply(lambda close: ta.EMA(close, h), axis=0)
        ema_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["EMA_{}".format(h)]])
        ema_df = pd.concat(objs=[ema_df, ema_tmp], axis=1).sort_index(axis=1)
        #scaled_ema = standardize(ema_df)
        
        # KAMA - Kaufman Adaptive Moving Average, http://www2.wealth-lab.com/WL5Wiki/(S(1htwupjlwryxzyjb4mrwaz45))/HTTrendLine.ashx
        kama_tmp = df_close.apply(lambda close: ta.KAMA(close, h), axis=0)
        kama_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["KAMA_{}".format(h)]])
        kama_df = pd.concat(objs=[kama_df, kama_tmp], axis=1).sort_index(axis=1)
        #scaled_kama = standardize(kama_df)
        
        # T3 - Triple Exponential Moving Average (T3)
        #df['T_3' + str(n)] = ta.T3(df['Close'], timeperiod=n, vfactor=.1)
        
        # WMA - Weighted Moving Average
        wma_tmp = df_close.apply(lambda close: ta.WMA(close, h), axis=0)
        wma_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["WMA_{}".format(h)]])
        wma_df = pd.concat(objs=[wma_df, wma_tmp], axis=1).sort_index(axis=1)
        #scaled_wma = standardize(wma_df)
        
        # CMO - Chande Momentum Oscillator
        cmo_tmp = df_close.apply(lambda close: ta.CMO(close, h), axis=0)
        cmo_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["CMO_{}".format(h)]])
        cmo_df = pd.concat(objs=[cmo_df, cmo_tmp], axis=1).sort_index(axis=1)
        #scaled_cmo = standardize(cmo_df)
        
        # MOM - Momentum
        mom_tmp = df_close.apply(lambda close: ta.MOM(close, h), axis=0)
        mom_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["MOM_{}".format(h)]])
        mom_df = pd.concat(objs=[mom_df, mom_tmp], axis=1).sort_index(axis=1)
        #scaled_mom = standardize(mom_df)
        
        # ROC - Rate of change : ((price/prevPrice)-1)*100
        roc_tmp = df_close.apply(lambda close: ta.ROC(close, h), axis=0)
        roc_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["ROC_{}".format(h)]])
        roc_df = pd.concat(objs=[roc_df, roc_tmp], axis=1).sort_index(axis=1)
        #scaled_roc = standardize(roc_df)
        
        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        rocp_tmp = df_close.apply(lambda close: ta.ROCP(close, h), axis=0)
        rocp_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["ROCP_{}".format(h)]])
        rocp_df = pd.concat(objs=[rocp_df, rocp_tmp], axis=1).sort_index(axis=1)
        #scaled_rocp = standardize(rocp_df)
        
        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        rocr100_tmp = df_close.apply(lambda close: ta.ROCR100(close, h), axis=0)
        rocr100_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["ROCR100_{}".format(h)]])
        rocr100_df = pd.concat(objs=[rocr100_df, rocr100_tmp], axis=1).sort_index(axis=1).astype(float)
        #scaled_rocr100 = standardize(rocr100_df)
        
        # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline, 
        ht_trendline_tmp = df_close.apply(ta.HT_TRENDLINE, axis=0)
        ht_trendline_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["HT_TRENDLINE_{}".format(h)]])
        ht_trendline_df = pd.concat(objs=[ht_trendline_df, ht_trendline_tmp], axis=1).sort_index(axis=1)
        #scaled_ht_trendline = standardize(ht_trendline_df)
    
        # ADX - Average Directional Movement Index
        # df ['adx' + str(n)] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=n)
        # ADXR - Average Directional Movement Index Rating
        # df['adxr' + str(n)] = ta.ADXR(df['High'], df['Low'], df['Close'], timeperiod=n)
        
        for ticker, ticker_df in df.groupby(axis=1, level=0):
            #open_serie = df.loc[:, (ticker, "Open")].to_numpy().squeeze()
            high_serie = df.loc[:, (ticker, "High")].to_numpy().squeeze()
            low_serie = df.loc[:, (ticker, "Low")].to_numpy().squeeze()
            close_serie = df.loc[:, (ticker, "Close")].to_numpy().squeeze()
            #volume = df.loc[:, (ticker, "Volume")].to_numpy().squeeze()
            
            # AROONOSC - Aroon Oscillator
            aroonosc_tmp = ta.AROONOSC(high_serie, close_serie, timeperiod=h)
            # Create DataFrame with the appropriate captions
            aroonosc_tmp = pd.DataFrame(aroonosc_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["ARRONOSC_{}".format(h)]]))
            # Concatenate temporary DataFrame for specific iteration into globall/cross-sectional DataFrame all over the iterations
            aroonosc_df = pd.concat(objs=[aroonosc_df, aroonosc_tmp], axis=1).sort_index(axis=1)
            #scaled_aroonosc = standardize(aroonosc_df)
            
            # CCI - Commodity Channel Index
            cci_tmp = ta.CCI(high_serie, low_serie, close_serie, timeperiod=h)
            cci_tmp = pd.DataFrame(cci_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["CCI_{}".format(h)]]))
            cci_df = pd.concat(objs=[cci_df, cci_tmp], axis=1).sort_index(axis=1)                                                                                                 
            #scaled_cci = standardize(cci_df)
            
            # DX - Directional Movement Index
            dx_tmp = ta.DX(high_serie, low_serie, close_serie, timeperiod=h)
            dx_tmp = pd.DataFrame(dx_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["DX_{}".format(h)]]))
            dx_df = pd.concat(objs=[dx_df, dx_tmp], axis=1).sort_index(axis=1)
            #scaled_dx = standardize(dx_df)
            
            '''
            # MFI - Money Flow Index
            File "_func.pxi", line 22, in talib._ta_lib.check_array
            Exception: input array type is not double
            mfi_tmp = ta.MFI(high_serie, low_serie, close_serie, volume, timeperiod=h)
            mfi_tmp = pd.DataFrame(mfi_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["MFI_{}".format(h)]]))
            mfi_df = pd.concat(objs=[mfi_df, mfi_tmp], axis=1).sort_index(axis=1)
            '''
            
            # MINUS_DI - Minus Directional Indicator
            minus_di_tmp = ta.MINUS_DI(high_serie, low_serie, close_serie, timeperiod=h)
            minus_di_tmp = pd.DataFrame(minus_di_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["MINUS_DI_{}".format(h)]]))
            minus_di_df = pd.concat(objs=[minus_di_df, minus_di_tmp], axis=1).sort_index(axis=1)
            #scaled_minus_di = standardize(minus_di_df)
            
            # PLUS_DI - Plus Directional Indicator
            plus_di_tmp = ta.PLUS_DI(high_serie, low_serie, close_serie, timeperiod=h)
            plus_di_tmp = pd.DataFrame(plus_di_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["PLUS_DI_{}".format(h)]]))
            plus_di_df = pd.concat(objs=[plus_di_df, plus_di_tmp], axis=1).sort_index(axis=1)
            #scaled_plus_di = standardize(plus_di_df)
                    
            # PLUS_DM - Plus Directional Movement
            plus_dm_tmp = ta.PLUS_DM(high_serie, low_serie, timeperiod=h)
            plus_dm_tmp = pd.DataFrame(plus_dm_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["PLUS_DM_{}".format(h)]]))
            plus_dm_df = pd.concat(objs=[plus_dm_df, plus_dm_tmp], axis=1).sort_index(axis=1)
            #scaled_plus_dm = standardize(plus_dm_df)
            
            # WILLR - Williams' %R
            willr_tmp = ta.WILLR(high_serie, low_serie, close_serie, timeperiod=h)
            willr_tmp = pd.DataFrame(willr_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["WILLR_{}".format(h)]]))
            willr_df = pd.concat(objs=[willr_df, willr_tmp], axis=1).sort_index(axis=1)
            #scaled_willr = standardize(willr_df)
            
            '''
            # AD - Chaikin A/D Line
            File "_func.pxi", line 22, in talib._ta_lib.check_array
            Exception: input array type is not double
            ad_tmp = ta.AD(high_serie, low_serie, close_serie, volume)
            ad_tmp = pd.DataFrame(ad_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["AD_{}".format(h)]]))
            ad_df = pd.concat(objs=[ad_df, ad_tmp], axis=1).sort_index(axis=1)
            '''
            # ATR - Average True Range
            atr_tmp = ta.ATR(high_serie, low_serie, close_serie, timeperiod=h)
            atr_tmp = pd.DataFrame(atr_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["ATR_{}".format(h)]]))
            atr_df = pd.concat(objs=[atr_df, atr_tmp], axis=1).sort_index(axis=1)
            #scaled_atr = standardize(atr_df)
            
             # NATR - Normalized Average True Range
            natr_tmp = ta.NATR(high_serie, low_serie, close_serie, timeperiod=h)
            natr_tmp = pd.DataFrame(natr_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["NATR_{}".format(h)]]))
            natr_df = pd.concat(objs=[natr_df, natr_tmp], axis=1).sort_index(axis=1)
            #scaled_natr = standardize(natr_df)
            
    trange_df = pd.DataFrame()
    
    for ticker, ticker_df in df.groupby(axis=1, level=0):
        #open_serie = df.loc[:, (ticker, "Open")].to_numpy().squeeze()
        high_serie = df.loc[:, (ticker, "High")].to_numpy().squeeze()
        low_serie = df.loc[:, (ticker, "Low")].to_numpy().squeeze()
        close_serie = df.loc[:, (ticker, "Close")].to_numpy().squeeze()
        #volume = df.loc[:, (ticker, "Volume")].to_numpy().squeeze()
        
        # TRANGE - True Range
        trange_tmp = ta.TRANGE(high_serie, low_serie, close_serie)
        trange_tmp = pd.DataFrame(trange_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["TRANGE"]]))
        trange_df = pd.concat(objs=[trange_df, trange_tmp], axis=1).sort_index(axis=1)
        #scaled_trange = standardize(trange_df)
    
    # POS - price position
    pos_df = price_position(df, horizons)     
            
    features = pd.concat(objs=[
        rsi_df, 
        ema_df, 
        kama_df, 
        wma_df, 
        cmo_df,
        mom_df,
        roc_df,
        rocp_df,
        rocr100_df,
        ht_trendline_df,
        aroonosc_df, 
        cci_df, 
        dx_df, 
        minus_di_df, 
        plus_di_df, 
        plus_dm_df, 
        willr_df, 
        atr_df, 
        natr_df, 
        trange_df,
        pos_df
        ], axis=1).sort_index(axis=1)    
       
    features = features.apply(pd.to_numeric, errors="coerce")
    
    return features

"""
# Example usage
df = read_yf_data(tickers=["AAPL", "FB", "AMZN", "MSFT", "IBM", "F", "CVX"])
horizons = [15, 22, 66]
features_technical = technical_factors(df, horizons)
"""
#%%
# Volume Technical Factors

def volume_technical_factors(df: pd.DataFrame, horizons: list):
    '''
    Function calculates volume technical indicators for specific horizons.
    
    Parameters
    ----------
    df: pandas.DataFrame
        Input DataFrame where is input data and where is the new feature added.
    
    Example
    -------
    >>>volume_technical_features = volume_technical_factors(data)
    ...
                           AAPL                         ...       MSFT                       
                       AAPL                ...          MSFT              
               Volume_CCI_10 Volume_CCI_22  ... Volume_WMA_22  Volume_WMA_5
    Date                                    ...                                                               ...                                  
                     ...           ...  ...           ...           ...
    2021-04-27   -139.123909   -168.277632  ...  2.476286e+07  2.491699e+07
    2021-04-28   -156.106064   -209.785120  ...  2.659100e+07  3.242624e+07
    2021-04-29    -34.175427    -82.605671  ...  2.778542e+07  3.630595e+07
    2021-04-30    207.982633    216.738361  ...  2.807914e+07  3.596687e+07
    2021-05-03    197.898288    288.768522  ...  2.736335e+07  3.121991e+07
    '''
    df_volume = df.loc[:, (slice(None), "Volume")]
    
    # All features declaration
    rsi_df = pd.DataFrame()
    ema_df = pd.DataFrame()
    kama_df = pd.DataFrame()
    wma_df = pd.DataFrame()
    cmo_df = pd.DataFrame()
    mom_df = pd.DataFrame()
    roc_df = pd.DataFrame()
    rocp_df = pd.DataFrame()
    rocr100_df = pd.DataFrame()
    ht_trendline_df = pd.DataFrame()
    cci_df = pd.DataFrame()
        
    for h in horizons:
        # RSI
        rsi_tmp = df_volume.apply(lambda volume: ta.RSI(volume, h))
        rsi_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_RSI_{}".format(h)]])
        rsi_df = pd.concat(objs=[rsi_df, rsi_tmp], axis=1).sort_index(axis=1)
        #scaled_rsi = standardize(rsi_df)
        
        # EMA - Exponential Moving Average
        ema_tmp = df_volume.apply(lambda volume: ta.EMA(volume, h), axis=0)
        ema_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_EMA_{}".format(h)]])
        ema_df = pd.concat(objs=[ema_df, ema_tmp], axis=1).sort_index(axis=1)
        #scaled_ema = standardize(ema_df)
        
        # KAMA - Kaufman Adaptive Moving Average, http://www2.wealth-lab.com/WL5Wiki/(S(1htwupjlwryxzyjb4mrwaz45))/HTTrendLine.ashx
        kama_tmp = df_volume.apply(lambda volume: ta.KAMA(volume, h), axis=0)
        kama_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_KAMA_{}".format(h)]])
        kama_df = pd.concat(objs=[kama_df, kama_tmp], axis=1).sort_index(axis=1)
        #scaled_kama = standardize(kama_df)
        
        # T3 - Triple Exponential Moving Average (T3)
        #df['T_3' + str(n)] = ta.T3(df['volume'], timeperiod=n, vfactor=.1)
        
        # WMA - Weighted Moving Average
        wma_tmp = df_volume.apply(lambda volume: ta.WMA(volume, h), axis=0)
        wma_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_WMA_{}".format(h)]])
        wma_df = pd.concat(objs=[wma_df, wma_tmp], axis=1).sort_index(axis=1)
        #scaled_wma = standardize(wma_df)
        
        # CMO - Chande Momentum Oscillator
        cmo_tmp = df_volume.apply(lambda volume: ta.CMO(volume, h), axis=0)
        cmo_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_CMO_{}".format(h)]])
        cmo_df = pd.concat(objs=[cmo_df, cmo_tmp], axis=1).sort_index(axis=1)
        #scaled_cmo = standardize(cmo_df)
        
        # MOM - Momentum
        mom_tmp = df_volume.apply(lambda volume: ta.MOM(volume, h), axis=0)
        mom_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_MOM_{}".format(h)]])
        mom_df = pd.concat(objs=[mom_df, mom_tmp], axis=1).sort_index(axis=1)
        #scaled_mom = standardize(mom_df)        
        
        # ROC - Rate of change : ((price/prevPrice)-1)*100
        roc_tmp = df_volume.apply(lambda volume: ta.ROC(volume, h), axis=0)
        roc_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_ROC_{}".format(h)]])
        roc_df = pd.concat(objs=[roc_df, roc_tmp], axis=1).sort_index(axis=1)
        #scaled_roc = standardize(roc_df)
        
        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        rocp_tmp = df_volume.apply(lambda volume: ta.ROCP(volume, h), axis=0)
        rocp_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_ROCP_{}".format(h)]])
        rocp_df = pd.concat(objs=[rocp_df, rocp_tmp], axis=1).sort_index(axis=1)
        #scaled_rocp = standardize(rocp_df)
        
        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        rocr100_tmp = df_volume.apply(lambda volume: ta.ROCR100(volume, h), axis=0)
        rocr100_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_ROCR100_{}".format(h)]])
        rocr100_df = pd.concat(objs=[rocr100_df, rocr100_tmp], axis=1).sort_index(axis=1)
        #scaled_rocr100 = standardize(rocr100_df)
        
        # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline, 
        ht_trendline_tmp = df_volume.apply(ta.HT_TRENDLINE, axis=0)
        ht_trendline_tmp.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_HT_TRENDLINE_{}".format(h)]])
        ht_trendline_df = pd.concat(objs=[ht_trendline_df, ht_trendline_tmp], axis=1).sort_index(axis=1)
        #scaled_ht_trendline = standardize(ht_trendline_df)
        
        # Commodity Channel Index https://www.investopedia.com/terms/c/commoditychannelindex.asp
        for ticker, ticker_df in df.groupby(axis=1, level=0):
            tp1_serie = df.loc[:, (ticker, "Volume")].shift(1).to_numpy().squeeze()
            tp2_serie = df.loc[:, (ticker, "Volume")].shift(2).to_numpy().squeeze()
            tp3_serie = df.loc[:, (ticker, "Volume")].shift(3).to_numpy().squeeze()
             
            cci_tmp = ta.CCI(tp1_serie, tp2_serie, tp3_serie, timeperiod=h)
            cci_tmp = pd.DataFrame(cci_tmp, index=df.index, columns=pd.MultiIndex.from_product(iterables=[[ticker], ["Volume_CCI_{}".format(h)]]))
            cci_df = pd.concat(objs=[cci_df, cci_tmp], axis=1).sort_index(axis=1)
            #scaled_cci = standardize(cci_df)
    
    # volume position - jiny vypocet nez price position: ne OHLC ale min/max z volume
    vol_pos_df = volume_position(df, horizons) 
    # pojmenovani features (aby pak bylo unikatni v ramci vsech features)
    #col_names = pos_df.columns.to_frame()
    #col_names.iloc[:, 1] = "Volume_" + col_names.iloc[:, 1] # prefix jmena v 1. levelu
    #pos_df.columns = pd.MultiIndex.from_frame(col_names)
    
    features = pd.concat(objs=[
            rsi_df, 
            ema_df, 
            kama_df, 
            wma_df, 
            cmo_df,
            mom_df,
            roc_df,
            rocp_df,
            rocr100_df,
            ht_trendline_df,
            cci_df,
            vol_pos_df
            ], axis=1).sort_index(axis=1)
    
    return features

#↨volume_technical_factors = volume_technical_factors(data, horizons)

#%%
def smoothed_volume_technical_factors(df: pd.DataFrame, horizon_smooth: int, horizons: list):
    """
    Computes technical features of smoothed volume

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with multilevel columns (ticker-indicator)
        
    horizon_smooth : int
        horizon for smoothing the volume series
        
    horizons : list
        lenght of horizons for rolling calculation

    """
    
    df_volume = df.loc[:, (slice(None), "Volume")]
    df_volume_smooth = df_volume.rolling(window=horizon_smooth).sum() # smoothed volume
    # pro pouziti v hotove fci to musi zustat pod sloupcem nazvanym "Volume" -> prejmenovat pak
    smoothed_vol_tech_factors = volume_technical_factors(df_volume_smooth, horizons)

    # kvuli POS (get_position) musi vstup obsahovat i OHLC sloupce, V nepouziva => muze byt tedy smoothed
    #df2 = df.copy() # pro jistotu
    #df2.loc[:, (slice(None), "Volume")] = df_volume_smooth # nemelo by prehazet poradi sloupcu
    #smoothed_vol_tech_factors = volume_technical_factors(df2, horizons)
    
    # prejmenovani features
    col_names = smoothed_vol_tech_factors.columns.to_frame()
    col_names.iloc[:, 1] = 'Smooth_{}_'.format(horizon_smooth) + col_names.iloc[:, 1] # jmena v 1. levelu
    smoothed_vol_tech_factors.columns = pd.MultiIndex.from_frame(col_names)
    
    return smoothed_vol_tech_factors

'''
horizons_smooth = 5
horizons = [15, 22, 66]
smoothed_volume_technical_factors(df, horizons_smooth, horizons)
'''
#%%
def standardized_unexplained_volume(df: pd.DataFrame, horizons: list): 
    """
    Computes firm-specific standardized unexplained volume and derived factors

    Parameters
    ----------
    df : pd.DataFrame
        input DataFrame with "raw" prices etc. data
        
    horizons : list
        lenght of horizons for rolling calculations

    Returns
    -------
    SUV_factors : pd.DataFrame
        DataFrame with firm-specific standardized unexplained volume etc.

    """
    # Standardized Unexplained Volume
    # https://www.biz.uiowa.edu/faculty/jgarfinkel/pubs/divop_JAR.pdf
    
    returns = compute_return(df)
    
    # Divide Returns into 2 separate columns (1st: return > 0 or 0, 2nd: return < 0 or 0)
    returns_positive = returns * (returns > 0)
    returns_positive.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Positive_Return"]])
    returns_negative = returns * (returns < 0)
    returns_negative.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Negative_Return"]])
    
    returns = pd.concat(objs=[returns_positive, returns_negative], axis=1).sort_index(axis=1)
    
    # Estimate linear regression
    SUV_coeff = pd.DataFrame()
    SUV_residual = pd.DataFrame()
    
    for ticker, ticker_df in returns.groupby(axis=1, level=0):
        
        # Prepare input data in sense: y=volume_serie, X=returns_tmp
        volume_serie = df.loc[:, (ticker, "Volume")]
        returns_df = ticker_df
        
        '''
        # Synchronize both datasets to the same index
        # https://stackoverflow.com/questions/48170867/how-to-get-the-common-index-of-two-pandas-dataframes
        common = volume_serie.index.intersection(returns_df.index) # common observations
        volume_serie = volume_serie.loc[common]
        returns_df = returns_df.loc[common]
        '''
        
        # Rolling equation estimation - temporary for horizons
        positive_coeff_ticker_specific = pd.DataFrame()
        negative_coeff_ticker_specific = pd.DataFrame()
        SUV_resid_firm = pd.DataFrame()
        SUV_coeff_firm = pd.DataFrame()
        
        for h in horizons: 
            
            ########## LinAlgError: Singular matrix
            try:
                """
                ##########################
                #STATMODELS ROLLING
                ##########################
                returns_tmp = sm.add_constant(returns_df, prepend=False)
                model = sm.regression.rolling.RollingOLS(volume_serie, returns_tmp, window=h, missing="drop")    
                rolling_results = model.fit() # snad zadny ticker a horizont neprobehne bez chyby
                ##########################
                #END of STATMODELS ROLLING
                ##########################
                """
                ##########################
                # PYFINANCE ROLLING
                ##########################
                rolling_results = ols.PandasRollingOLS(y=volume_serie, x=returns_df, window=h)
                ##########################
                # END of PYFINANCE ROLLING
                ##########################
            except:
                print("Rolling OLS error in horizon {} of ticker {} -> NaNs for whole coef and resid columns".format(h, ticker))
                # prazdny sloupce kdyz nejde provest ols odhad
                std_last_residual_h_specific = pd.DataFrame(data=np.nan, index=volume_serie.index, 
                                                            columns=["SUV_Std_err_last_resid_{}".format(h)])
                coeff_firm_h_spec = pd.DataFrame(data=np.nan, index=volume_serie.index, columns=["SUV_Coef_{}".format(h)])
            else:
                ##########################
                # Results
                ##########################
    
                # Betas for Positive Returns
                positive_coeff_h_specific = rolling_results.beta.iloc[:, 1]
                positive_coeff_h_specific.name = "Beta_Positive_Returns_{}".format(h)
                positive_coeff_ticker_specific = pd.concat(objs=[positive_coeff_ticker_specific, positive_coeff_h_specific], axis="columns")
                
                # Betas for Positive Returns
                negative_coeff_h_specific = rolling_results.beta.iloc[:, 0]
                negative_coeff_h_specific.name = "Beta_Negative_Returns_{}".format(h)
                negative_coeff_ticker_specific = pd.concat(objs=[negative_coeff_ticker_specific, negative_coeff_h_specific], axis="columns")
                
                # Standard Error
                standard_error_h_specific = rolling_results.std_err.replace(to_replace=0, value=np.nan) # vyhodit pripadne nuly kvuli deleni dale
                
                # Residuals
                rolling_residuals = rolling_results.resids.unstack(level="subperiod")
                last_residual_h_specific = pd.Series(data=None, index=rolling_residuals.index, dtype="float")
                for idx in rolling_residuals.index: # for each specific rolling window (row)
                    last_residual_h_specific.loc[idx] = rolling_residuals.loc[idx, idx] # from time serie take the last observation and drop nan
                    # TO DO: VECTORIZE
                std_last_residual_h_specific = last_residual_h_specific/standard_error_h_specific 
                std_last_residual_h_specific.name = "SUV_Std_err_last_resid_{}".format(h)
                #SUV_resid_firm = pd.concat(objs=[SUV_resid_firm, std_last_residual_h_specific], axis="columns")
                
                coeff_firm_h_spec = (positive_coeff_h_specific - negative_coeff_h_specific) / (standard_error_h_specific/np.sqrt(h))
                #SUV_Coef(kStart:end,j,e)=(B(:,2)-B(:,3))./(B(:,end-2)./sqrt(p));
                coeff_firm_h_spec.name = "SUV_Coef_{}".format(h)
            
            SUV_resid_firm = pd.concat(objs=[SUV_resid_firm, std_last_residual_h_specific], axis="columns")
            SUV_coeff_firm = pd.concat(objs=[SUV_coeff_firm, coeff_firm_h_spec], axis="columns")
            
        # priradit ke konkretnimu tickeru
        SUV_resid_firm.columns = pd.MultiIndex.from_product(iterables=[[ticker], SUV_resid_firm.columns])
        SUV_coeff_firm.columns = pd.MultiIndex.from_product(iterables=[[ticker], SUV_coeff_firm.columns])
        
        # Standardized Unexplained Volume
        SUV_residual = pd.concat(objs=[SUV_residual, SUV_resid_firm], axis="columns")
        SUV_coeff = pd.concat(objs=[SUV_coeff, SUV_coeff_firm], axis="columns")
        
    # Aggregated residuals
    SUV_residual_aggregated = pd.DataFrame()
    for h in [5, 10, 22, 22*2, 22*3]: 
        SUV_residual_agg = SUV_residual.rolling(window=h, axis="index").sum()
        SUV_residual_agg = SUV_residual_agg.stack(level=0).add_suffix("_cum_{}".format(h)).unstack(level=1) # suffix v nazvu feature
        SUV_residual_agg = SUV_residual_agg.swaplevel(i=0, j=1, axis="columns").sort_index(axis="columns") # tickery v nejvyssim levelu (problem s nazvy)

        SUV_residual_aggregated = pd.concat(objs=[SUV_residual_aggregated, SUV_residual_agg], axis="columns") 
        
    #SUV_residual_aggregated = SUV_residual_aggregated.swaplevel(i=0, j=1, axis="columns").sort_index(axis="columns") # prohodit vse najednou (problem s nazvy)
    
    SUV_factors = pd.concat(objs=[SUV_residual, SUV_residual_aggregated, SUV_coeff], axis="columns")
    SUV_factors.index = pd.to_datetime(SUV_factors.index)
    
    return SUV_factors


def standardized_unexplained_volume_weeks(df: pd.DataFrame, horizons: list): 
    """
    Computes firm-specific standardized unexplained volume and derived factors in weeks variant

    Parameters
    ----------
    df : pd.DataFrame
        input DataFrame with "raw" prices etc. data
        
    horizons : list
        lenghts of horizons for rolling calculations

    Returns
    -------
    suv_factors_w : pd.DataFrame
        DataFrame with firm-specific standardized unexplained volume weeks etc.

    """
    
    # Convert price and volume time series to weekly frequency
    df_resampled_w = df.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index") # first -> vytvori se NaNy
    idx_resampled = df_resampled_w.index.day_of_week==4 # patek - ma v pripadne neobchodniho dne vyplnene predchozi hodnoty
    df_resampled_w = df_resampled_w.loc[idx_resampled] # jake hodnoty plati k tomu danemu konci obdobi (i kdyz neni business day)

    suv_factors_w = standardized_unexplained_volume(df_resampled_w, horizons)
    suv_factors_w = suv_factors_w.stack(level=0).add_suffix("_weeks").unstack(level=1) # suffix v nazvu feature
    suv_factors_w = suv_factors_w.swaplevel(i=0, j=1, axis="columns").sort_index(axis="columns") # tickery v nejvyssim levelu
    suv_factors_w = suv_factors_w.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index") # first -> vytvori se NaNy
    idx = set(suv_factors_w.index).intersection(set(df.index)) # jen dny, ktere jsou i v zakladnich datech - vyhozeni vikendu atd.
    suv_factors_w = suv_factors_w.loc[idx, :].sort_index(axis="index")
    # na konci muzou chybet dny pokud posledni den v dennich zdrojovych datech (df) nebyl patek - pocita se s df_resampled_w, kde jsou jen patky
    idx_last = df.loc[df_resampled_w.index[-1]:, :].index # od posledniho v df_resampled_w do posledniho v df
    suv_factors_w_last = pd.DataFrame(data=np.nan, index=idx_last, columns=suv_factors_w.columns)
    if len(suv_factors_w_last)>1: # delka jen 1 = neni nic noveho (zacatek na poslednim pozorovani v df, coz uz ale v suv_factors_w je)
        suv_factors_w = pd.concat(objs=[suv_factors_w, suv_factors_w_last.iloc[1:]], # od 2. pozorovani
                                   axis="index").fillna(method="ffill", axis="index") 
    
    return suv_factors_w


def ar1_coef(x_data: pd.Series):
    """
    Calculation of beta coefficient from auto-regression

    Parameters
    ----------
    x_data : pd.Series
        numeric series from which autoregression will be performed

    Returns
    -------
    ar1_beta : float
        beta coefficient from auto-regression of order 1

    """
    try: 
        # kdyz jsou samy nany - "samo" to vrati NaN
        ar_mdl = sm.tsa.arima.model.ARIMA(endog=x_data, order=(1, 0, 0)).fit()
    except: # kdyby se nahodou nenafitoval model napr. kvuli singular matrix
        ar1_beta = np.nan
    else:
        ar1_beta = ar_mdl.arparams[0]
        
    return ar1_beta

    
def ar1_factors(df: pd.DataFrame, horizons: list):
    """
    Computes firm-specific rolling auto-regression beta coefficient of log. adj. close

    Parameters
    ----------
    df : pd.DataFrame
        input DataFrame with "raw" prices etc. data
        
    horizons : list
        lenghts of horizons for rolling calculations
        
    Returns
    -------
    ar_factors : pd.DataFrame
        DataFrame with firm-specific rolling auto-regression beta coefficients

    """

    logclose = np.log(df.loc[:, (slice(None), "Close")].droplevel(level=1, axis="columns"))#[400:600]
    logclose_raw = pd.DataFrame(data=logclose.to_numpy(), columns=logclose.columns) # bez indexu
    # s indexem je pak v ARIMA warning: A date index has been provided... 
    
    ar_factors = pd.DataFrame()
    for h in horizons: 
        x_ar1 = logclose_raw.rolling(window=h, axis="index").aggregate(ar1_coef) # rolling AR coefs for all tickers
        x_ar1.index = df.index # vraceni puvodniho indexu
        x_ar1.columns = pd.MultiIndex.from_product(iterables=[logclose.columns, ["AR1_LogPrice_{}".format(h)]])
        
        ar_factors = pd.concat(objs=[ar_factors, x_ar1], axis="columns").sort_index(axis="columns")
        print("Horizon ", h, " completed")
    
    return ar_factors

#ar1_logprice = ar1_factors(data, horizons)


def ar1_factors_weeks(df: pd.DataFrame, horizons: list): 
    """
    Computes firm-specific rolling auto-regression beta coefficient of log. adj. close in weeks variant

    Parameters
    ----------
    df : pd.DataFrame
        input DataFrame with "raw" prices etc. data
        
    horizons : list
        lenghts of horizons for rolling calculations

    Returns
    -------
    ar1_logprice_w : pd.DataFrame
        DataFrame with firm-specific rolling auto-regression beta coefficients in weeks variant

    """
    
    # Convert price and volume time series to weekly frequency
    df_resampled_w = df.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index") # first -> vytvori se NaNy
    idx_resampled = df_resampled_w.index.day_of_week==4 # patek - ma v pripadne neobchodniho dne vyplnene predchozi hodnoty
    df_resampled_w = df_resampled_w.loc[idx_resampled] # jake hodnoty plati k tomu danemu konci obdobi (i kdyz neni business day)

    ar1_logprice_w = ar1_factors(df_resampled_w, horizons) # strasne dlouho trva vypocet !!!
    ar1_logprice_w = ar1_logprice_w.stack(level=0).add_suffix("_weeks").unstack(level=1) # suffix v nazvu feature
    ar1_logprice_w = ar1_logprice_w.swaplevel(i=0, j=1, axis="columns").sort_index(axis="columns") # tickery v nejvyssim levelu
    ar1_logprice_w = ar1_logprice_w.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index") # first -> vytvori se NaNy
    idx = set(ar1_logprice_w.index).intersection(set(df.index)) # jen dny, ktere jsou i v zakladnich datech - vyhozeni vikendu atd.
    ar1_logprice_w = ar1_logprice_w.loc[idx, :].sort_index(axis="index")

    # na konci muzou chybet dny pokud posledni den v dennich zdrojovych datech (df) nebyl patek - pocita se s df_resampled_w, kde jsou jen patky
    idx_last = df.loc[df_resampled_w.index[-1]:, :].index # od posledniho v df_resampled_w do posledniho v df
    ar1_logprice_w_last = pd.DataFrame(data=np.nan, index=idx_last, columns=ar1_logprice_w.columns)
    if len(ar1_logprice_w_last)>1: # delka jen 1 = neni nic noveho (zacatek na poslednim pozorovani v df, coz uz ale v ar1_logprice_w je)
        ar1_logprice_w = pd.concat(objs=[ar1_logprice_w, ar1_logprice_w_last.iloc[1:]], # od 2. pozorovani
                                   axis="index").fillna(method="ffill", axis="index") 
    
    return ar1_logprice_w


#%%
##########################
# Fundamental Data
##########################
def get_fundamentals(tickers, horizons, dir_basic):
    '''
    Returns firm-specific fundamental data, change in fundamental data and specific relative position of the fundamental data.
    
    Parameters
    ----------
    tickers : list
        List of tickers.
    horizons : list 
        List of horizons for rolling windows
        
    Returns
    -------
        DataFrame with firm-specific fundamental data and derived predictors.

    '''
    
    ##########################
    # Load Data
    ##########################
    #♠scaled_fundamentals = pd.DataFrame()
    fundamentals_raw = pd.DataFrame()
    
    for ticker in tickers:
        path = os.path.join(os.getcwd(), "../SP-data/fundamental_data/firm_specific/src_fundamental/{}_daily.parquet".format(ticker))
        #path = os.path.join(dir_basic, "fundamental_data/firm_specific/daily_data/{}_daily.parquet".format(ticker)) # Why is it better? Relative path is defined
        try:
            fundamental_ticker_specific = pd.read_parquet(path)
        except:
            print("Fundamental data - parquet file for ", ticker, " not found at ", path)
            fundamental_ticker_specific = pd.DataFrame()
        fundamental_ticker_specific  = fundamental_ticker_specific.apply(pd.to_numeric, errors="coerce")
        col_names = ["_".join(col_tup) for col_tup in fundamental_ticker_specific.columns] # 1-levelove col_names
        fundamental_ticker_specific.columns = pd.MultiIndex.from_product(iterables=[[ticker], col_names]) # NotImplementedError: isna is not defined for MultiIndex 
        fundamentals_raw = pd.concat(objs=[fundamentals_raw, fundamental_ticker_specific], axis="columns")
        # nejen u FB nastava InvalidIndexError: Reindexing only valid with uniquely valued Index objects 
        
    fundamentals_raw = fundamentals_raw.loc[fundamentals_raw.index.notna()] # vyhodit NaT/NaN v indexu
    
    #serie_names = fundamentals_raw.columns.levels[1] # names of predictors for other manipulation
    # pocty prediktoru nejsou stejne pro kazdy ticker => vzit jen to co je spolecne pro vsechny tickery
    # .. jinak by dale vzniklo ValueError: Length mismatch: Expected axis has 5 elements, new values have 7 elements
    try: # kdyby fundamentals_raw byl prazdny df (sloupce by nebyly multilevel)
        serie_names = set(fundamentals_raw.columns.levels[1]) # maximum prediktoru
    except: 
        serie_names = set()
        fundamentals = pd.DataFrame()
    else:
        for ticker in tickers:
            ticker_data = fundamentals_raw.loc[:, (ticker, slice(None))]
            # v columns.levels[1] jsou vsechny puvodni sloupce (ne vsechny v podvyberu jsou), ale v columns jen ty "spravne"
            ticker_factors = pd.unique( pd.DataFrame(data=list(ticker_data.columns))[1] )
            #print(ticker, len(ticker_factors))
            serie_names = serie_names.intersection( set(ticker_factors) )
    
        serie_mis = set(fundamentals_raw.columns.levels[1]) - serie_names
        print(len(serie_mis), " fundamental factors are discarded - unavailable for every asset): \n", serie_mis)
    
        fundamentals = fundamentals_raw.loc[:, (slice(None), list(serie_names))]
        
        # kontrola dat - same nany
        data_valid = pd.DataFrame()
        for serie_name in serie_names:
            serie = fundamentals.loc[:, (slice(None), serie_name)]
            serie_valid = serie.notna().sum()
            serie_valid.index = serie_valid.index.droplevel(level=1)
            serie_valid.name = serie_name
            data_valid = pd.concat(objs=[data_valid, serie_valid], axis="columns").sort_index(axis="columns")
        
        vars_invalid = data_valid.sum()==0
        vars_invalid = list(vars_invalid[vars_invalid].index) 
        print(len(vars_invalid), " fundamental factors are discarded - completely not numbers or missing: \n", set(vars_invalid) )
        fundamentals = fundamentals.drop(labels=vars_invalid, axis="columns", level=1)
        
        serie_names = serie_names - set(vars_invalid) # jen validni ukazatele
    
    ##########################
    # Changes and Positions
    ##########################
    
    for serie_name in serie_names:
        serie = fundamentals.loc[:, (slice(None), serie_name)]
        # scaling outstanding fundamental data - DEPRECATED!
        # scaled_fundamentals_tmp = standardize(serie)
        # scaled_fundamentals = pd.concat(objs=[scaled_fundamentals, scaled_fundamentals_tmp], axis="columns")
        
        for h in horizons:            
            # horizon specific change
            change = serie.sub(serie.shift(periods=h, axis="index")) # horizon specific change
            change.columns = pd.MultiIndex.from_product(iterables=[tickers, [serie_name + "_Change_{}".format(h)]]) # TypeError: can only concatenate tuple (not "str") to tuple
            # scaled_change = standardize(change)
            
            # horizon specific position
            position_higher_bound = serie.rolling(window=h).max() 
            position_lower_bound = serie.rolling(window=h).min() 
            # bohuzel deleni 0 nevyrobi +/-inf ale ZeroDivisionError: float division by zero
            position_denom = (position_higher_bound.sub(position_lower_bound)).replace(to_replace=0, value=np.nan) # vyhodit nuly
            position = (serie.sub(position_lower_bound)).divide( position_denom ) # nany tam, kde by se delilo nulou
            position.columns = pd.MultiIndex.from_product(iterables=[tickers, [serie_name + "_Position_{}".format(h)]])
            #scaled_position = standardize(position)
            
            fundamentals = pd.concat(objs=[fundamentals, change, position], axis="columns").sort_index(axis="columns")
        #print("All derived predictors calculated for ", serie_name)
    
    return fundamentals

#fundamental_test = get_fundamentals(tickers, horizons)

#%%0
##########################
# Price Position
##########################
def price_position(df, horizons):
    """
    Returns firm-specific price position through specific horizons.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame where is input data.
    horizons : list
        List of horizons for rolling windows.

    Returns
    -------
    position: pandas.DataFrame
        DataFrame with firm-specific price position

    """
    position = pd.DataFrame()
    
    high_serie = df.loc[:, (slice(None), "High")]
    low_serie = df.loc[:, (slice(None), "Low")]
    current_close = df.loc[:, (slice(None), "Close")]
        
    for h in horizons:    
        highest_high = high_serie.rolling(window=h).max().to_numpy()
        lowest_low = low_serie.rolling(window=h).min().to_numpy()
                
        # Relative position of the price (firm-specific)
        # deleni nulou by nemelo nastat (shodly by se highest_high a lowest_low)
        position_h_denom = np.subtract(highest_high, lowest_low)
        position_h_denom[position_h_denom==0] = np.nan # vyhodit nuly
        
        position_h_specific = np.subtract(current_close, lowest_low) / position_h_denom #_h_specific is a temporary dataframe for current horizon        
        position_h_specific.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Price_Position_{}".format(h)]])
        #scaled_position = standardize(position_h_specific)
        
        position = pd.concat(objs=[position, position_h_specific], axis="columns").sort_index(axis="columns")
        
    return position

#position_test = price_position(data, horizons)
#%%

def volume_position(df: pd.DataFrame, horizons: list):
    """
    Computes firm-specific volume position in specific horizons.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (V is enough) with multilevel columns (ticker-indicator)
        
    horizons : list
        horizons for rolling windows

    Returns
    -------
    vol_position : pd.DataFrame
        firm-specific volume position

    """    
    vol_serie = df.loc[:, (slice(None), "Volume")]
    
    vol_position = pd.DataFrame()
    for h in horizons:    
        highest_high = vol_serie.rolling(window=h).max()#.to_numpy()
        lowest_low = vol_serie.rolling(window=h).min()#.to_numpy()
                
        # Relative position of the price (firm-specific)
        
        # preventivni reseni deleni nulou
        position_h_denom = np.subtract(highest_high, lowest_low).replace(to_replace=0, value=np.nan) # vyhodit nuly
        position_h_specific = np.subtract(vol_serie, lowest_low) / position_h_denom 
        position_h_specific.columns = pd.MultiIndex.from_product(iterables=[df.columns.levels[0], ["Volume_Position_{}".format(h)]])
        
        vol_position = pd.concat(objs=[vol_position, position_h_specific], axis="columns").sort_index(axis="columns")
        
    return vol_position



#%% Flatten Hiearchical Structured Data

#df.columns = df.columns.map('_'.join).str.strip()
#print(df)









