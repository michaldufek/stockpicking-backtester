#import yfinance as yf
import pandas as pd
import numpy as np

#from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import scipy.stats
from statsmodels.stats.proportion import proportions_chisquare_allpairs
from empyrical import max_drawdown


def quantile_stats(predictor: pd.Series, target: pd.Series, nQuant: int):
    """
    QuantilePD2 - Matlab

    "Replacement" (not real replacement) of predictor values with quantile stats
    (quantile maximum of factor, quantile mean of target)
    
    predictor : pd.Series
        one observation (day, etc.) of predictor for all assets
    
    target : pd.Series
        one observation (day, etc.) of target for all assets
    
    nQuant : int
        number of quantiles the predictor will be splitted to
        
    Returns
    -------
    q_info : pd.DataFrame 
        properties of individual quantiles
    
    factor_info : pd.DataFrame
        original and replaced predictor and target values with quantile properties
        
    """
    
    factor_info = predictor.copy()
    quantiles = np.linspace(start=0, stop=1, num=nQuant+1)
    # olabelovani kvantilu dle faktoru (inputu): list(range(1, nQuant+1)) / False (cisla 0–nQuant)
    rnd = False # vyskyt "ridkych" dat (nenaplnene kvantily)

    try:
        q_assign, q_bounds = pd.qcut(factor_info, quantiles, labels=list(range(1, nQuant+1)), retbins=True, duplicates="drop")
    except ValueError:
        # ValueError: Bin labels must be one fewer than the number of bin edges
        # ... vznikly by prazdne kvantily
        #q_assign, q_bounds = pd.qcut(factor_info, quantiles, labels=None, retbins=True, duplicates="drop")
        # k faktoru se pridaji nahodne mala cisla, aby do prazdnych kv. neco "padlo"
        random_add = pd.Series(list(range(1, len(predictor)+1)))*(10**(-10))
        random_add.index = predictor.index # aby slo scitat na pozicich dle tickeru
        factor_info += random_add
        q_assign, q_bounds = pd.qcut(factor_info, quantiles, labels=list(range(1, nQuant+1)), retbins=True, duplicates="drop")
        # prazdne kvantily by uz vznikat nemely, tj. nepusobit chybu
        rnd = True # kdy/jak casto toto nastalo
        # nebo mozna bude stacit to pustit bez "drop duplicates" - defaultne "raise"
        #q_assign, q_bounds = pd.qcut(factor_info, quantiles, labels=list(range(1, nQuant+1)), retbins=True)
    
    # prazdne kvantily uz nevzniknou, resp. na vystupu by nemely byt
    factor_info = pd.concat(objs=[factor_info, q_assign], axis="columns")
    factor_info.columns = ["val_factor", "q_label"]
    
    # informacni prehled o kvantilech
    q_labels = sorted(list(pd.unique(q_assign.dropna())))
    q_info = pd.DataFrame(data=q_labels, index=q_labels, columns=["q_label"])
    
    # nahrazeni faktoru maximalnimi kvantilovymi hodnotami faktoru
    q_attr = factor_info.groupby("q_label").max()["val_factor"].to_dict()
    factor_info["factor_qmax"] = factor_info["q_label"].replace(q_attr)
    q_info["q_factor_max"] = q_info["q_label"].replace(q_attr) # doplneni info o kvantilech
    
    # nahrazeni faktoru prumernymi kvantilovymi hodnotami targetu/outputu
    target_info = target.copy()
    target_info.name = "val_target"
    factor_info = pd.concat(objs=[factor_info, target_info], axis="columns")
    q_attr = factor_info.groupby("q_label").mean()["val_target"].to_dict() # kvantilovy prumer targetu
    factor_info["target_qmean"] = factor_info["q_label"].replace(q_attr)
    
    # doplneni info o kvantilech
    q_info["q_target_mean"] = q_info["q_label"].astype(float).replace(q_attr) # nejak se sem dostanou nany????
    # asi nefunguje replace - resp vraci nany
    # mozna vadi, ze q_label nejsou cisla ale asi stringy......
    #q_info["q_target_mean"] = 99 # POKUS !!!!! cislo natvrdo "funguje" - na vystupu all_factors_mimicking_returns nejsou samy nany
    q_attr = factor_info["q_label"].value_counts().to_dict()
    q_info["q_count"] = q_info["q_label"].replace(q_attr)
    
    factor_info[["factor_qmax", "target_qmean"]] = factor_info[["factor_qmax", "target_qmean"]]#.fillna(0)
    
    return q_info, factor_info, rnd

"""
# EXAMPLE usage of quantile_stats
factor_all = features_all.loc[:, (slice(None), "CurrencyVolume_5")].shift(periods=-1, axis="index")
factor_all.columns = factor_all.columns.droplevel(level=1) # nechat v columns jen level s tickerama
target_all = features_all.loc[:, (slice(None), "Return")]
target_all.columns = target_all.columns.droplevel(level=1) # nechat v columns jen level s tickerama

nQuant = 3

predictor = factor_all.loc[factor_all.index[30-1], :]
output = target_all.loc[target_all.index[30], :]

quant_info, quantiled_data, rnd = quantile_stats(predictor, output, nQuant)
"""



def quantile_stats_insample(predictor: pd.Series, target: pd.Series, nQuant: int, outsample_start: str):
    """
    QuantilePD2_Out - Matlab

    "Replacement" (not real replacement) of predictor values with quantile stats computed on in-sample data partition
    (quantile maximum of factor, quantile mean of target)
        
    predictor : pd.Series
        evolution of one feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
    
    target : pd.Series
        evolution of target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    nQuant : int
        number of quantiles the predictor will be splitted to
        
    outsample_start : str
        start date of out-sample partition of data
        
    Returns
    -------
    q_info : pd.DataFrame 
        properties of individual quantiles
    
    factor_info : pd.DataFrame
        original and replaced predictor and target values with quantile properties
        
    """
    
    factor_insample = predictor.loc[:, :outsample_start].copy() # v in-sample nebude posledni datum
    factor_info = factor_insample.copy()

    quantiles = np.linspace(start=0, stop=1, num=nQuant+1)
    # olabelovani kvantilu dle insample casti faktoru (inputu): list(range(1, nQuant+1)) / False (cisla 0–nQuant)
    
    #_, q_bounds = pd.qcut(factor_insample, quantiles, labels=list(range(1, nQuant+1)), retbins=True) 
    # mohl by byt pak stejny problem s pripadnyma prazdnyma kvantilama jako u quantile_stats()    
    rnd = False # vyskyt "ridkych" dat (nenaplnene kvantily)
    try: # Ficura ale osetreni "ridkych" dat zde nema narozdil od podobne situace a QuantilePD2() uvnitr StockPicking_FM_Returns() 
        _, q_bounds = pd.qcut(factor_insample, quantiles, labels=list(range(1, nQuant+1)), retbins=True, duplicates="drop")
    except ValueError:
    
        # ValueError: Bin labels must be one fewer than the number of bin edges
        # ... vznikly by prazdne kvantily
        #q_assign, q_bounds = pd.qcut(factor_info, quantiles, labels=None, retbins=True, duplicates="drop")
        # k faktoru se pridaji nahodne mala cisla, aby do prazdnych kv. neco "padlo"
        random_add = pd.Series(list(range(1, len(factor_insample)+1)))*(10**(-10))
        random_add.index = factor_insample.index # aby slo scitat na pozicich dle tickeru
        factor_info += random_add
        q_assign, q_bounds = pd.qcut(factor_info, quantiles, labels=list(range(1, nQuant+1)), retbins=True, duplicates="drop")
        # prazdne kvantily by uz vznikat nemely, tj. nepusobit chybu
        rnd = True # kdy/jak casto toto nastalo
        # nebo mozna bude stacit to pustit bez "drop duplicates" - defaultne "raise"
        #q_assign, q_bounds = pd.qcut(factor_info, quantiles, labels=list(range(1, nQuant+1)), retbins=True)
        
    # upraveni hranic krajnich intervalu (neomezene otevreni) aby se mohlo olabelovat vse v out-samplu resp. full-samplu
    q_bounds[0] = -np.inf #predictor.min()-1
    q_bounds[-1] = np.inf #predictor.max()+1
    q_assign, _ = pd.cut(predictor, q_bounds, labels=list(range(1, nQuant+1)), retbins=True) 
    
    # mergnout s celyma datama - prazdne kvantily nevzniknou, resp. na vystupu nejsou (je jich min) -> resit jinde
    factor_info = pd.concat(objs=[predictor, q_assign], axis="columns")
    factor_info.columns = ["val_factor", "q_label"]
    
    # informacni prehled o kvantilech
    q_labels = pd.unique(q_assign.dropna())
    q_info = pd.DataFrame(data=q_labels, index=q_labels, columns=["q_label"])
    
    # nahrazeni faktoru maximalnimi kvantilovymi hodnotami faktoru
    q_attr = factor_info.groupby("q_label").max()["val_factor"].to_dict()
    factor_info["factor_qmax"] = factor_info["q_label"].replace(q_attr)
    q_info["q_factor_max"] = q_info["q_label"].replace(q_attr) # doplneni info o kvantilech
    
    # nahrazeni faktoru prumernymi kvantilovymi hodnotami targetu/outputu
    target_info = target.copy()
    target_info.name = "val_target"
    factor_info = pd.concat(objs=[factor_info, target_info], axis="columns")
    q_attr = factor_info.groupby("q_label").mean()["val_target"].to_dict() # kvantilovy prumer targetu
    factor_info["target_qmean"] = factor_info["q_label"].replace(q_attr)
    
    # doplneni info o kvantilech
    q_info["q_target_mean"] = q_info["q_label"].replace(q_attr) 
    q_attr = factor_info["q_label"].value_counts().to_dict()
    q_info["q_count"] = q_info["q_label"].replace(q_attr)
    
    factor_info[["factor_qmax", "target_qmean"]] = factor_info[["factor_qmax", "target_qmean"]].fillna(0)
    
    return q_info.sort_index(axis="index"), factor_info



def each_observation_transformation(factor: pd.Series, target: pd.Series, nQuant: int):
    """
    Factor transformation as quantile target mean

    Parameters
    ----------
    factor : pd.Series
        one observation (day, etc.) of predictor for all assets
        
    target : pd.Series
        one observation (day, etc.) of target for all assets
        
    nQuant : int
        number of quantiles the factor will be splitted to

    Returns
    -------
    fact_mim_target : pd.Series 
        factor values transformed into quantile target mean values
        
    fact_mim_corr : float
        spearman rank correlation of factor (at t-1) and subsequent target (at t)
        
    rnd : bool
        indication of randomization when the data cannot be splitted into nQuant quantiles

    """
    #fact_mim_target = pd.Series(0, index=list(range(nQuant)), name="q_target_mean")
    fact_mim_target = pd.Series(0, index=list(range(1, nQuant+1)), name="q_target_mean")
    fact_mim_corr = 0
    
    # vstupem do kvantilove transformace budou rady s 1-level indexem
    predictor = factor.copy()
    output = target.copy()
    
    rnd = False # vyskyt "ridkych" dat (nenaplnene kvantily)

    nStocks = predictor.notna().multiply(output.notna()).sum() # pocet aktivnich assetu
    if nStocks>0:
        quant_info, _, rnd = quantile_stats(predictor, output, nQuant)
        """
        # prazdne kvantily nevzniknou, resp. na vystupu nejsou (je jich min) -> resit
        if len(quant_info) < nQuant: # pokud nejake kvantily byly prazdne
            # k faktoru se pridaji nahodne mala cisla, aby do prazdnych kv. neco "padlo"
            random_add = pd.Series(list(range(1, len(predictor)+1)))*(10**(-10))
            random_add.index = predictor.index # aby slo scitat na pozicich dle tickeru
            predictor += random_add
            quant_info, _ = quantile_stats(predictor, output, nQuant)
            rnd = True # kdy/jak casto toto nastalo
        """
        # ulozeni prumernych kvantilovych hodnot vsech kvantilu
        fact_mim_target = quant_info["q_target_mean"].astype(float) # FM_QuantRet(i,:)

        fact_mim_corr = scipy.stats.spearmanr(a=predictor, b=output, nan_policy="omit")[0]
        #predictor.corr(other=output, method="spearman")
    
    return fact_mim_target, fact_mim_corr, rnd


"""
# EXAMPLE usage of each_observation_transformation
factor_all = features_all.loc[:, (slice(None), "CurrencyVolume_5")].shift(periods=-1, axis="index")
factor_all.columns = factor_all.columns.droplevel(level=1) # nechat v columns jen level s tickerama
target_all = features_all.loc[:, (slice(None), "Return")]
target_all.columns = target_all.columns.droplevel(level=1) # nechat v columns jen level s tickerama

nQuant = 3

factor_day = factor_all.loc[factor_all.index[3], :]
target_day = target_all.loc[target_all.index[3], :]

fm_target, fm_corr, rand = each_observation_transformation(factor_day, target_day, nQuant)
"""



def factor_mimicking_returns(factor_all: pd.DataFrame, target_all: pd.DataFrame, nQuant: int):
    """
    StockPicking_FM_Returns - Matlab
    
    Factor-mimicking portfolio returns (for one factor): returns recalculated by factor quantiles

    Parameters
    ----------
    factor_all : pd.DataFrame
        one feature for all assets in portfolio, same dimensions as returns
        
    returns_all : pd.DataFrame
        returns for all assets in portfolio, same dimensions as factor
        
    nQuant : int
        number of quantiles the factor will be splitted to        
        
    Returns
    -------
    df_qtarget : pd.DataFrame
        evolution of all quantiles' values (table with nQuant columns)
    
    df_qtarget_cum : pd.DataFrame
        cummulative evolution of all quantiles' values (table with nQuant columns)
    
    df_qtarget_diff : pd.Series
        evolution of difference between best and worse quantiles' values
    
    df_qtarget_diff_cum : pd.Series
        cummulative evolution of difference between best and worse quantiles' values
    
    df_corr : pd.Series
        evolution of spearman correlation of factor (at t-1) and subsequent target (at t)
    
    nRand : int
        number of randomizations when the data couldn't be splitted into nQuant quantiles
    """
    
    factor_all = factor_all.copy().shift(periods=1, axis="index") # faktor v case t-1, copy pro jistotu

    df_qtarget = pd.DataFrame(dtype=float)
    #df_corr = pd.DataFrame(data=0, columns=["corr"], index=factor_all.index)
    df_corr = pd.Series(data=0, name="corr", index=factor_all.index, dtype=float)
    df_corr = df_corr.astype(float)
    nRand = 0
    for ( index_date, factor_day ), ( _, target_day ) in zip( factor_all.iterrows(), target_all.iterrows() ) :
        quant_ret, fm_corr, rand = each_observation_transformation(factor_day, target_day, nQuant)
        
        # kvantilove prumery targetu
        quant_ret.name = index_date
        quant_ret = quant_ret.to_frame().T # series nejde transponovat
        df_qtarget = pd.concat(objs=[df_qtarget.astype(float), quant_ret.astype(float)], axis="index") 
        
        # korelace vcerejsiho faktoru a dnesniho targetu (returnu)
        #df_corr.loc[index_date, "corr"] = float(fm_corr)
        df_corr.loc[index_date] = float(fm_corr)
        
        # pocet vyskytu nenaplnenych kvantilu
        nRand += rand
        #print("nRand",nRand)
    
    # rozdily nejlepsiho a nejhorsiho kvantilu
    df_qtarget_diff = df_qtarget.iloc[:, -1] - df_qtarget.iloc[:, 0]
    
    # + kumulativni kvantilove prumery, rozdily nejlepsiho a nejhorsiho kvantilu
    df_qtarget_cum = df_qtarget.cumsum(axis=0).astype(float)
    df_qtarget_diff_cum = df_qtarget_diff.cumsum(axis=0).astype(float)
    
    return df_qtarget, df_qtarget_cum, df_qtarget_diff.astype(float), df_qtarget_diff_cum, df_corr, nRand


"""
# EXAMPLE usage of factor_mimicking_returns

factor_all = features_all.loc[:, (slice(None), "CurrencyVolume_5")]
factor_all.columns = factor_all.columns.droplevel(level=1) # nechat v columns jen level s tickerama
target_all = features_all.loc[:, (slice(None), "Return")]
target_all.columns = target_all.columns.droplevel(level=1) # nechat v columns jen level s tickerama

nQuant = 3

# FM returns pro 1 faktor (pro vsechny assety)
fm_ret, fm_ret_cum, fm_ret_diff, fm_ret_diff_cum, fm_corr, nRand = factor_mimicking_returns(factor_all, target_all, nQuant)

# transformovane returny nemusi vzdy byt serazene (tak jak jsou faktory, podle nichz se tranformuje)
sum(fm_ret_cum[1]>fm_ret_cum[2])
sum(fm_ret_cum[2]>fm_ret_cum[3])
sum( (fm_ret_cum[1]>fm_ret_cum[2]) & (fm_ret_cum[2]>fm_ret_cum[3]) )
"""



def all_factors_mimicking_returns(all_features: pd.DataFrame, target_all: pd.DataFrame, nQuant: int, verbose: bool = False):
    """
    StockPicking_FM_Returns_All - Matlab
    
    Factor-mimicking portfolio returns for all factors

    Parameters
    ----------
    all_features : pd.DataFrame
        all features for all assets in portfolio (nAssets x nFeatures columns)
        
    target_all : pd.DataFrame
        returns for all assets in portfolio (nAssets columns) - must be one-level columns' names!!!
        
    nQuant : int
        number of quantiles the features will be splitted to
    
    verbose: bool
        indicator for printing processing information (no printing in default)

    Returns
    -------
    port_fm_ret_diff : pd.DataFrame
        evolution of difference between best and worse quantiles' values for each factor
        
    port_fm_ret_diff_cum : pd.DataFrame
        cummulative evolution of difference between best and worse quantiles' values for each factor
        
    port_fm_ret : pd.DataFrame
        evolution of all quantiles' values for each factor (table with nQuant x nFactors columns)
        
    port_fm_ret_cum : pd.DataFrame
        cummulative evolution of all quantiles' values for each factor (table with nQuant x nFactors columns)
        
    port_fm_corr : pd.DataFrame
        evolution of spearman correlation of factor (at t-1) and subsequent target (at t) for each factor
        
    port_nrand : pd.Series
        number of randomizations when the data couldn't be splitted into nQuant quantiles for each factor

    """
    
    port_fm_ret_diff = pd.DataFrame(dtype=float)
    port_fm_ret_diff_cum = pd.DataFrame(dtype=float)
    port_fm_corr = pd.DataFrame(dtype=float)
    port_fm_ret = pd.DataFrame(dtype=float)
    port_fm_ret_cum = pd.DataFrame(dtype=float)
    
    # v columns.levels[1] jsou vsechny puvodni sloupce, ale v columns jen ty "spravne"
    factors = pd.unique( pd.DataFrame(data=list(all_features.columns))[1] )
    port_nrand = pd.Series(data=0, name="n_rnd", index=factors)

    for f in factors:
        if verbose:
            print(f)  
        factor = all_features.loc[:, (slice(None), f)]
        factor.columns = factor.columns.droplevel(level=1) # nechat v columns jen level s tickerama
    
        fm_ret, fm_ret_cum, fm_ret_diff, fm_ret_diff_cum, fm_corr, nRand = factor_mimicking_returns(factor, target_all, nQuant)
    
        # jednosloupcove vystupy
        fm_ret_diff.name = f
        port_fm_ret_diff = pd.concat(objs=[port_fm_ret_diff, fm_ret_diff], axis="columns")
        fm_ret_diff_cum.name = f
        port_fm_ret_diff_cum = pd.concat(objs=[port_fm_ret_diff_cum, fm_ret_diff_cum], axis="columns")
        fm_corr.name = f
        port_fm_corr = pd.concat(objs=[port_fm_corr, fm_corr], axis="columns")
        
        # vicesloupcove vystupy - sloupce: faktory x kvantily
        fm_ret.columns = pd.MultiIndex.from_product(iterables=[[f], list(fm_ret.columns) ])
        port_fm_ret = pd.concat(objs=[port_fm_ret, fm_ret], axis="columns")
        fm_ret_cum.columns = pd.MultiIndex.from_product(iterables=[[f], list(fm_ret_cum.columns) ])
        port_fm_ret_cum = pd.concat(objs=[port_fm_ret_cum, fm_ret_cum], axis="columns")
    
        # jednoradkovy vystup
        port_nrand.loc[f] = nRand
        if verbose:
            print(" -- OK")
    
    return port_fm_ret_diff, port_fm_ret_diff_cum, port_fm_ret, port_fm_ret_cum, port_fm_corr, port_nrand



def portfolio_perf_stats(port_fmret_diff_cum: pd.DataFrame, port_fmret_cum: pd.DataFrame, port_fmcorr: pd.DataFrame, y_gran: int):
    """
    StockPicking_FM_PerformanceStats - Matlab
    
    Compute performance statistics of factor-mimicking portfolios returns

    Parameters
    ----------
    port_fmret_diff_cum : pd.DataFrame
        cumul. evol. of differences between best and worst quantiles of factor-mimicking portfolios returns
        
    port_fmret_cum : pd.DataFrame
        cumul. evol. of all quantiles of factor-mimicking portfolios returns
        
    port_fmcorr : pd.DataFrame
        evolution of spearman correlation of factor and subsequent target for each factor
        
    y_gran : int
        yearly granularity - number of periods/observations in a year

    Returns
    -------
    stats : pd.DataFrame
        performance statistics of factor-mimicking portfolios returns

    """
    
    stats = pd.DataFrame()
    
    ## Cumulative returns of factor-mimicking portfolios
    # odecteni prvni hodnoty => zacatek na 0 (Ficura to nejdriv diferencuje, pak kumuluje)
    fmret_basic = port_fmret_diff_cum-port_fmret_diff_cum.loc[port_fmret_diff_cum.index[0], :]
    # Position in the FMP (1=High-Low, -1=Log-High)
    fmret_direction = np.sign(fmret_basic.loc[fmret_basic.index[-1], :])
    fmret_direction.name = "direction"
    stats = pd.concat(objs=[stats, fmret_direction], axis="columns")
    
    kPeriods = (fmret_basic!=0).sum(axis="index")
    
    # annual return
    stat_tmp = ((fmret_direction * fmret_basic.loc[fmret_basic.index[-1], :])/kPeriods) * y_gran
    stat_tmp.name = "return_pa"
    stats = pd.concat(objs=[stats, stat_tmp], axis="columns")
    
    # maximum drawdown
    fmret_tmp = fmret_direction * fmret_basic
    fmret_mdd = max_drawdown(fmret_tmp)
    fmret_mdd.index = fmret_basic.columns
    fmret_mdd.name = "max_dd"
    # RuntimeWarning: overflow encountered in multiply
    # RuntimeWarning: invalid value encountered in subtract
    # zaporny nuly tam nevadej: fmret_tmp[fmret_tmp==-0] = 0
    stats = pd.concat(objs=[stats, fmret_mdd], axis="columns")
    
    # drawdown ratio
    stat_tmp = (fmret_direction * fmret_basic.loc[fmret_basic.index[-1], :])/(-fmret_mdd)
    stat_tmp.name = "dd_ratio"
    stats = pd.concat(objs=[stats, stat_tmp], axis="columns")
    
    # t-stat a t-pval
    fmret_tstat, fmret_tpval = scipy.stats.ttest_1samp(a=fmret_tmp, popmean=0, axis=0, nan_policy="omit", alternative="two-sided")
    fmret_tstat = pd.Series(data=fmret_tstat, index=fmret_basic.columns, name="T_stat")
    fmret_tpval = pd.Series(data=fmret_tpval, index=fmret_basic.columns, name="T_pval")
    stats = pd.concat(objs=[stats, fmret_tstat, fmret_tpval], axis="columns")
    
    ## Cumulative returns in the long-only portfolio
    fmret_basic = pd.DataFrame()
    # odecteni prvni hodnoty => zacatek na 0 (Ficura to nejdriv diferencuje, pak kumuluje)
    fmret_qvals_basic = port_fmret_cum-port_fmret_cum.loc[port_fmret_cum.index[0], :]
    # pro kazdy faktor vzit hodnoty prislusneho kvantilu dle direction
    for factor, factor_qvals in fmret_qvals_basic.groupby(axis="columns", level=0):
        fmret_qvals_tmp = factor_qvals.sort_index(axis="columns", level=1) # pro jistotu
        if fmret_direction.loc[factor]>0: # hodnoty nejvyssiho kvantilu faktoru
            fmret_tmp = fmret_qvals_tmp.iloc[:, -1]
        elif fmret_direction.loc[factor]<0: # hodnoty nejnizsiho kvantilu faktoru
            fmret_tmp = fmret_qvals_tmp.iloc[:, 0]
        fmret_tmp.name = factor
        fmret_basic = pd.concat(objs=[fmret_basic, fmret_tmp], axis="columns")
    
    kPeriods = (fmret_basic!=0).sum(axis="index")
    kPeriods.name = "n_obs"
    
    # long-only portfolio annual return
    stat_tmp = (fmret_basic.loc[fmret_basic.index[-1], :]/kPeriods) * y_gran
    stat_tmp.name = "lport_return_pa"
    stats = pd.concat(objs=[stats, stat_tmp], axis="columns")
    
    # long-only portfolio maximum drawdown
    fmret_mdd = max_drawdown(fmret_basic) # opet overflow encountered in multiply, invalid value encountered in subtract
    fmret_mdd.index = fmret_basic.columns
    fmret_mdd.name = "lport_max_dd"
    stats = pd.concat(objs=[stats, fmret_mdd], axis="columns")
    
    # long-only portfolio drawdown ratio
    stat_tmp = fmret_basic.loc[fmret_basic.index[-1], :]/(-fmret_mdd)
    stat_tmp.name = "lport_dd_ratio"
    stats = pd.concat(objs=[stats, stat_tmp], axis="columns")
    
    # long-only portfolio t-stat a t-pval
    fmret_tmp = fmret_basic.diff(periods=1, axis="index")
    fmret_tstat, fmret_tpval = scipy.stats.ttest_1samp(a=fmret_tmp, popmean=0, axis=0, nan_policy="omit", alternative="two-sided")
    fmret_tstat = pd.Series(data=fmret_tstat, index=fmret_basic.columns, name="lport_T_stat")
    fmret_tpval = pd.Series(data=fmret_tpval, index=fmret_basic.columns, name="lport_T_pval")
    stats = pd.concat(objs=[stats, fmret_tstat, fmret_tpval], axis="columns")
    
    ## Spearman correlation between factor and returns
    fmcorr_basic = port_fmcorr.replace(to_replace=0, value=np.nan) # dale jen nenulove korelace
    
    # correlation medians
    stat_tmp = fmcorr_basic.median(axis="index", skipna=True)
    stat_tmp.name = "corr_med"
    stats = pd.concat(objs=[stats, stat_tmp], axis="columns")
    stat_tmp = abs(stat_tmp)
    stat_tmp.name = "corr_med_abs"
    stats = pd.concat(objs=[stats, stat_tmp], axis="columns")
    
    # t-test applied to the Spearman correlation
    fmret_tstat, fmret_tpval = scipy.stats.ttest_1samp(a=fmcorr_basic, popmean=0, axis=0, nan_policy="omit", alternative="two-sided")
    fmret_tstat = pd.Series(data=fmret_tstat, index=fmret_basic.columns, name="corr_T_stat")
    fmret_tpval = pd.Series(data=fmret_tpval, index=fmret_basic.columns, name="corr_T_pval")
    stats = pd.concat(objs=[stats, fmret_tstat, fmret_tpval], axis="columns")
    
    ## Number of observations
    stats = pd.concat(objs=[stats, kPeriods], axis="columns")
    
    return stats



def remove_correlated(predictors_stacked: pd.DataFrame, predictors_perf: pd.Series, maxCorr: float):
    """
    Correlation_Remove_Lin - Matlab
    
    Filter out the predictors simultaneously having high correlation with other predictors and lower performance

    Parameters
    ----------
    predictors_stacked : pd.DataFrame
        evolution of all features for all assets in portfolio in long/stacked form (nDays*nAssets rows x nFeatures columns)
        
    predictors_perf : pd.Series
        performance (p-val etc.) of each of the predictors
        
    maxCorr : float
        maximum tolerated correlation among predictors

    Returns
    -------
    predictors_out : pd.DataFrame
        filtered predictors (in long/stacked form)
        
    vars_keep : list
        names of kept predictors
        
    predictors_corr : pd.DataFrame
        corrected correlation matrix of predictors (high correlations of underperforming predictors are set to 0)

    """

    predictors_corr = predictors_stacked.corr(method="pearson") # NaNy jsou ignorovane
    
    # "vyhozeni" vysokych korelaci sama se sebou - "label-based" diagonal filling 
    #idx_diag = predictors_corr.index.intersection(other=predictors_corr.columns) # klice existujici v indexu i columns
    #diag_vals = pd.Series(0, index=pd.MultiIndex.from_arrays(arrays=[idx_diag, idx_diag])) # hodnoty a jejich souradnice
    #predictors_corr.stack(dropna=False).update(diag_vals) # Ficura ale nechava vysoke kor. sama se sebou... ????
    # pro kazdou promennou: z promennych "sama sebe" a prip. dalsich s vysokyma korelacema vybere tu s nejvyssi performance a ponecha ji
    # u ostatnich s vysokyma korelacema je odstrani
    
    high_corr = abs(predictors_corr)>maxCorr
    
    vars_keep = list(predictors_stacked.columns) # na zacatku vsechny ponechat
    
    if high_corr.sum().sum()>len(high_corr): # kdyz se vyskytla vysoka korelace i mimo diagonalu
        #corr_vars = high_corr.index[high_corr.any()] # promenne s necim vysoce korelujici v pripade nul na hlavni diagonale
        corr_vars = predictors_stacked.columns # vzdy je vysoka korelace min. sama se sebou
        # pro kazdou promennou s necim vysoce korelujici nechat jen tu s nejvetsi p-hodnotou
        for corr_var in corr_vars:
            corr_vars_pvals = predictors_perf.loc[high_corr.loc[corr_var, :]] # p-hodnoty promennych s necim vysoce korelujicich
            best_predictor = corr_vars_pvals.idxmax(axis=0) # prediktor s nejlepsim performance
            predictors_corr.loc[corr_var, best_predictor] = 0 # oznaceni jako nizke korelace => ponechat
            #predictors_corr.loc[best_predictor, corr_var] = 0 # zrcadlove -"-
    
        #high_corr_final = abs(predictors_corr)>maxCorr
        low_corr_final = abs(predictors_corr)<maxCorr
        vars_keep = low_corr_final.all(axis="index") # low korelace musi byt se vsema
        vars_keep = list(vars_keep.index[vars_keep]) # aby fce vracela vzdy stejny typ (list vs. index. vs. series)
    
    predictors_out = predictors_stacked.loc[:, vars_keep]

    return predictors_out, vars_keep, predictors_corr



def univar_linreg_stats(X: pd.Series, Y: pd.Series):
    """
    My_LinReg - Matlab
    
    Univariate (continuous target) OLS regression with scalar outputs

    Parameters
    ----------
    X : pd.Series
        predictor values (treated as continuous)
        
    Y : pd.Series
        target values (treated as continuous)

    Returns
    -------
    b : float
        estimated beta_1/slope
        
    t : float
        t-statistic for a beta_1 parameter estimate
        
    p : float
        two-tailed p-value for the t-stat of the beta_1
        
    r2 : float
        adjusted R-squared of the model

    """
    X_mat = sm.add_constant(X) # ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
    result = sm.OLS(Y, X_mat).fit()
    
    # vystupy skalary
    b = result.params.iloc[1] # beta_1 - samotne cislo, jinak to pri apply vrati matici s betama na diagonale
    t = result.tvalues.iloc[1] # t-statistika beta_1
    p = result.pvalues.iloc[1] # p-hodnota beta_1
    r2 = result.rsquared_adj # adjusted R2
    
    # vystupy pole - nejsou potreba, resp. neni vhodne pro zakladni pouziti pri apply
    #Yf = result.fittedvalues 
    #resid = result.resid 
    
    return b, t, p, r2



def WoE(count_pos: int, count_neg: int, min_pos: int, min_neg: int, power_pos: float, power_neg: float):
    """
    Compute weight of evidence of certain quantile based conditions and (precomputed) quantile characteristics.
    Categories that do not fullfill the min_pos and min_neg conditions are removed from the sample and their WoE is set to 0
    
    Parameters
    ----------
    count_pos : int
        number of positive values/returns in quantile
        
    count_neg : int
        number of nonpositive values/returns in quantile
        
    min_pos : int
        minimal number of positive values/returns in quantile
        
    min_neg : int
        minimal number of nonpositive values/returns in quantile
        
    power_pos : float
        positive power of quantile
        
    power_neg : float
        negative/nonpositive power of quantile

    Returns
    -------
    WoE : float
        computed quantile weight of evidence

    """

    WoE_val = 0 # tam, kde nejsou splneny pozadavky na variabilitu kvantilu (pocet + a - returnu)
    if (count_pos>=min_pos) & (count_neg>=min_neg):
        WoE_val = np.log(power_neg)-np.log(power_pos) # mira prevazujici vlastnosti (+ => ztratovost; - => ziskovost)
    
    return WoE_val



def quantile_stats_categoric(predictor: pd.Series, target_bin: pd.Series):
    """
    DefaultPD - Matlab (similar to QuantilePD2_Out/quantile_stats but on quantiled data = without number of quantiles as input)

    "Replacement" (not real replacement) of predictor values with quantile stats (hit rate)
    
    predictor : pd.Series
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    Returns
    -------
    q_info : pd.DataFrame 
        properties of individual quantiles
    
    q_data : pd.DataFrame
        original and replaced predictor and target values with quantile properties
        
    """

    tmp_data = pd.concat(objs=[predictor, target_bin], axis="columns")
    #tmp_data = tmp_data.dropna(axis="index", how="any") # pro jistotu kdyby nebyly vstupy stejne dlouhe
    #.. aby nebyly nany v prediktorech a nevznikaly NaNove uniq hodnoty
    tmp_data.columns = ["predictor", "target"]
    
    # informacni prehled o kvantilech
    q_labels = sorted(pd.unique(predictor))
    q_info = pd.DataFrame(data=q_labels, index=q_labels, columns=["q_label"])
    
    # info o velikosti kvantilu
    q_attr = tmp_data.groupby("predictor").count()["target"] # pripadne NaNy v targetu jsou vynechane (defaultne)
    q_attr.name = "q_count"
    q_info = pd.concat(objs=[q_info, q_attr], axis="columns")
    
    # info o poctu cilovych hodnot (kladnych returnu) v kvantilu
    q_attr = tmp_data.groupby("predictor").sum()["target"] # pripadne NaNy v targetu jsou vynechane
    q_attr.name = "q_hit_count"
    q_info = pd.concat(objs=[q_info, q_attr], axis="columns")
    
    # info o podilu cilovych hodnot (uspesnost kvantilu)
    q_info["q_hit_rate"] = q_info["q_hit_count"] / q_info["q_count"]
    
    # nahrazeni prediktoru prumernymi kvantilovymi uspesnostmi (podily cilovych hodnot v targetu/outputu)
    q_attr = q_info["q_hit_rate"].to_dict() # kvantilova uspesnost targetu
    tmp_data = tmp_data.drop(labels="target", axis="columns")
    #tmp_data["target_qmean"] = tmp_data["predictor"]
    tmp_data["target_qmean"] = tmp_data["predictor"].replace(q_attr)
    
    return q_info, tmp_data



def quantiles_success_test(predictor: pd.Series, target_bin: pd.Series):
    """
    Categories_Binomial_Test - Matlab

    Calculates quantiles' stats (hit rate etc.) and quantiles relationships
    
    predictor : pd.Series
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    Returns
    -------
    quant_info : pd.DataFrame 
        properties of individual quantiles
    
    bin_test_mat : pd.DataFrame
        matrix of quantiles dissimilarities represented by 1-pvalues from chi-squared test for proportions
        
    """
    
    # DefaultPD - z kateg. dat vyextrahuje kategorie a jejich hit rate
    quant_info, _ = quantile_stats_categoric(predictor, target_bin)
    
    # Chi-Squared Test for Proportions? Is there any significant difference between (2 or more) survival proportions?
    # => matice: vzajemne vztahy kategorii/kvantilu - vyznamnost rozdilu v uspesnostech kvantilu
    counts = quant_info["q_hit_count"].to_numpy() # list, series to nebere -TypeError: list indices must be integers or slices, not list
    nobs = quant_info["q_count"].to_numpy()
    result_prop = proportions_chisquare_allpairs(count=counts, nobs=nobs, multitest_method="hs") # default Holm step down correction using Sidak adjustments
    # https://towardsdatascience.com/an-overview-of-methods-to-address-the-multiple-comparison-problem-310427b3ba92
    # musi byt vice nez 1 skupina, jinak ValueError: zero-size array to reduction operation maximum which has no identity
    pval_mat = result_prop.pval_table() # jen horni trojuhelnik, 
    # .pval_table() ...FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`.
    pval_mat_mirror = np.rot90(np.fliplr(pval_mat)) # zrcadleni
    pval_mat = 1 - (result_prop.pval_table() + pval_mat_mirror) # => soumerna matice, ma to byt 1-pval a ne jen pval ???????
    #pval_mat = result_prop.pval_table() + pval_mat_mirror # => soumerna matice, ma to byt 1-pval a ne jen pval ???????
    # pro pouziti by stacila nesoumerna, ale lepsi je dal nespolehat na to, ktery trojuhelnik bude vyplneny a ktery prazdny
    np.fill_diagonal(pval_mat, 0) # shoda = uplna nerozdilnost sama se sebou
    
    bin_test_mat = pd.DataFrame(data=pval_mat, index=quant_info["q_count"].index, columns=quant_info["q_count"].index)
    
    return quant_info, bin_test_mat



def category_woeization_predictor(predictor_cat: pd.Series, target_bin: pd.Series, min_pos: int, min_nonpos: int):
    """
    Category_Woeization - Matlab
    
    "Replacement" (not real replacement, but new column) of (factor) quantile values by quantile WoE

    Parameters
    ----------
    predictor_cat : pd.Series
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    min_pos : int
        minimal number of positive values/returns in quantile
        
    min_nonpos : int
        minimal number of negative/nonpositive values/returns in quantile

    Returns
    -------
    quantiled_factor : pd.DataFrame
        "enriched" predictor data with quantile WoE
        
    quantiles_info : pd.DataFrame
        "enriched" properties of individual quantiles with WoE etc.
        
    IV_predictor : float
        information value of the predictor (cumulative woe of all quantiles)
        
    """
    
    # vytahnout z kategorickych dat info o kvantilech
    quantiles_info, quantiled_factor = quantile_stats_categoric(predictor_cat, target_bin) # nevadi ze predictor je delsi nez target
    # delsi predictor nez target => vzniknou NaNy u poslednich dni kazdeho assetu, kde nejsou targety -> dale pak chyba
    # obohaceni kvantilovych statistik (vytazenych z dat)
    quantiled_factor["tgt_positive"] = (target_bin==1).astype(int) # s NaNy v bool sloupci (bez .astype(int)) groupby/sum ten sloupec ignoruje
    #KeyError: "None of [Index(['tgt_positive'], dtype='object')] are in the [columns]"
    q_info = quantiled_factor.groupby("predictor").sum()[["tgt_positive"]]
    quantiles_info = pd.concat(objs=[quantiles_info, q_info], axis="columns")
    quantiles_info["tgt_negative"] = quantiles_info["q_count"] - quantiles_info["tgt_positive"]
    quantiles_info["cond_ok"] = (quantiles_info["tgt_positive"]>=min_pos) & (quantiles_info["tgt_negative"]>=min_nonpos)
    
    # propsani do kvantilovanych dat - nahrazeni prediktoru priznakem splneni variability kvantilu
    q_attr = quantiles_info["cond_ok"].to_dict() # splneni podminek variability
    quantiled_factor["cond_ok"] = quantiled_factor["predictor"].replace(q_attr)
    
    # spocitani WoE a IV - hodnoty jen u kvantilu splnujicich min. variabilitu, pro neOK kvantily bude WoE=0
    # cDef = pomer poctu klad. ret. v kvantilu a poctu klad. ret. ve vsech dostatecne variabilnich kvantilech
    quantiles_info["q_power_positive"] = quantiles_info["tgt_positive"] / ( quantiled_factor["cond_ok"] & (quantiled_factor["tgt_positive"]==1) ).sum() 
    # cNoDef = pomer poctu zapor. ret. v kvantilu a poctu zapor. ret. ve vsech dostatecne variabilnich kvantilech
    quantiles_info["q_power_negative"] = quantiles_info["tgt_negative"] / ( quantiled_factor["cond_ok"] & ~(quantiled_factor["tgt_positive"]==1) ).sum() 
    # po zmene quantiled_factor["tgt_positive"] z bool na int nefunguje "~" jako negace -> TypeError -> radsi zmeneno i vyse u "q_power_positive"
    #TypeError: ufunc 'invert' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
    # .. deleni nulou by nemelo nastat, kvuli minimalnim pozadavkum na variabilitu
    # .. pripadne poresit nahrazeni +inf/-inf hodnot treba nulou
    quantiles_info["WoE"] = quantiles_info.\
            apply(lambda row: WoE(row["tgt_positive"], row["tgt_negative"], min_pos, min_nonpos, row["q_power_positive"], row["q_power_negative"] ), 
                                                 axis="columns", result_type="expand")
    
    IV_tmp = quantiles_info["WoE"] * (quantiles_info["q_power_negative"] - quantiles_info["q_power_positive"])
    IV_predictor = IV_tmp.sum() # information value - vaha faktoru (kumulovana za vsechny kvantily)
    
    # nahradit v datech hodnoty kvantilu prislusnym kvantilovym WoE
    q_woe = quantiles_info[["q_label", "WoE"]].set_index("q_label")["WoE"].to_dict()
    quantiled_factor["q_woe"] = quantiled_factor["predictor"].replace(q_woe) # doplneni info o kvantilech
    quantiled_factor = quantiled_factor.drop(labels=["cond_ok", "tgt_positive"], axis="columns") # vyhozeni "pracovnich" mezivypoctu
    
    return quantiled_factor, quantiles_info, IV_predictor



def chi2_test(predictor_cat: pd.Series, target_bin: pd.Series):
    """
    Tests impact of predictor categorization on category/quantile success rate

    Parameters
    ----------
    factor_cat : pd.Series
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)

    Returns
    -------
    chi2pval : float
        p-value of the chi-squared test (independence of the successfulness in the categories/quantiles)

    """
    #from sklearn.feature_selection import chi2
    #chi_stat, chi_pval = chi2(quantiled_factor["q_label"].to_frame(), target_bin) # target_qmean => ValueError: Input X must be non-negative.    
    cont_table = pd.crosstab(index=predictor_cat, columns=target_bin).sort_index(axis="index")
    #chi2stat, chi2pval = scipy.stats.chisquare(f_obs=cont_table, axis=1) # za kazdy kvantil zvlast
    _, chi2pval, _, _ = scipy.stats.chi2_contingency(observed=cont_table) # za vsechny kvantily (lze vratit i ocekavane cetnosti)
    # https://stats.stackexchange.com/questions/2391/what-is-the-relationship-between-a-chi-squared-test-and-test-of-equal-proportion

    return chi2pval



def univar_analysis_stats(predictor: pd.Series, target_num: pd.Series, target_bin: pd.Series, nQ: int):
    """
    Univariate_Analysis_v2 - Matlab
    
    Compute performance statistics of one predictor - from analysis of impact of predictor on target variable
    
    Parameters
    ----------
    predictor : pd.Series
        evolution of one feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_num : pd.Series
        evolution of target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)

    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    nQ: int
        number of quantiles the predictor will be splitted to

    Returns
    -------
    b : float
        estimated beta_1/slope
        
    p : float
        two-tailed p-value for the t-stat of the beta_1
         
    r2 : float
        adjusted R-squared of the model
        
    IV_predictor : float
        information value of the predictor (cumulative woe of all quantiles)
        
    chi2pval : float
        p-value of the chi-squared test (independence of the successfulness in the categories/quantiles)

    """

    min_pos, min_nonpos = 1, 1 # min. variabilita pro woeizaci: pocet klad. a pocet neklad. returnu v kvantilu
    
    ## regrese - sila vlivu numericke varianty prediktoru na numericky vysledek
    b, t, p, r2 = univar_linreg_stats(predictor, target_num)
    
    ## kategorizace - nahrazeni hodnot prediktoru kvantilovymi prumery targetu
    _, quantiled_factor, _ = quantile_stats(predictor, target_num, nQ)
    predictor_cat = quantiled_factor["target_qmean"]

    ## woeizace na kvantilovanych (kategorickych) prediktorech
    quantiled_factor, quantiles_info, IV_predictor = category_woeization_predictor(predictor_cat, target_bin, min_pos, min_nonpos) 
    
    # QuickGini - vystupy z toho predane dale (AR) jako vystup Univariate_Analysis_v2 nepouzity v "hlavnim" skriptu
    # .. vstupy jsou: bin. target a kvantilovany ("target_qmean") prediktor
    
    ## ChiSquaredTest - test vlivu kategorizace na uspesnost v kvantilu
    chi2pval = chi2_test(quantiled_factor["target_qmean"], target_bin) # asi nevadi kdyz je predictor delsi nez target_bin

    # ulozit statistiky do vystupu
    return b, p, r2, IV_predictor, chi2pval



def factor_qmax_transformation(predictor_stacked: pd.Series, target_stacked: pd.Series, nQuant: int, outsample_start: str):
    """
    Wrapper for quantile_stats_insample output usage - replacing of predictor values with quantile maximums

    Parameters
    ----------
    predictor_stacked : pd.Series
        evolution of one feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
    
    target_stacked : pd.Series
        evolution of target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
    
    nQuant : int
        number of quantiles the predictor will be splitted to
        
    outsample_start : str
        start date of out-sample partition of data
        
    Returns
    -------    
    factor_qmax : pd.Series
        predictor values replaced with quantile maximums

    """
    
    # "uceni" na in-sapmlu, pouziti na celych datech
    _, quantiled_factor = quantile_stats_insample(predictor_stacked, target_stacked, nQuant, outsample_start) 
    
    return quantiled_factor["factor_qmax"]



def category_merging_predictor(predictor_cat: pd.Series, target_bin: pd.Series, pConf: float, outsample_start: str):
    """
    Binning_PD_Ordinal_Out - Matlab
    
    Evaluates similarity of categories/quantiles and merge them when necessary

    Parameters
    ----------
    predictor_cat : pd.Series
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    pConf : float
        maximal similarity of categories for not joining them - p-value from test
        
    outsample_start : str
        start date of out-sample partition of data

    Returns
    -------
    predictor_cat_all : pd.Series
        predictor values with merged quantiles
        
    quant_info_in, quant_info_out, quant_info_all : pd.DataFrame
        properties of individual quantiles for in-, out- and whole sample of data
        
    """
    
    predictor_cat_in = predictor_cat.loc[:, :outsample_start].copy() # v in-sample nebude posledni datum
    predictor_cat_out = predictor_cat.loc[:, outsample_start:].copy() 
    
    target_bin_in = target_bin.loc[:, :outsample_start].copy() 
    target_bin_out = target_bin.loc[:, outsample_start:].copy() 
    
    pMax=1 # inicializace
    n_categ = len(pd.unique(predictor_cat_in))
    while (pMax>pConf) and (n_categ>1): # zatimco existuji prilis podobne kategorie/kvantily a v in-samplu se neposlucovalo do jednoho
        # maximalni existujici podobnost je vetsi nez nejvyssi tolerovana podobnost
        quant_info, quant_relations = quantiles_success_test(predictor_cat_in, target_bin_in) # Categories_Binomial_Test - ma tam byt 1-pval a ne jen pval ???????
        # quant_relations [poc. kvantilu x poc. kvantilu] - miry vzaj. rozdilnosti uspesnosti (target_bin) kategorii/kvantilu 
        # quant_info [poc. kvantilu x 4 charakteristiky] - oznaceni kvantilu (cislo), pocet hodnot, pocet uspechu, uspesnost
        # vztahy sama se sebou uz jsou potlaceny
        
        # misto 1 diagonaly nad i pod hlavni diagonalou by stacilo vyextrahovat jen 1 z nich => sousedni/nejpodobnejsi kvantily
        mat_1 = np.tril(m=np.ones(shape=quant_relations.shape, dtype=int), k=1) # vyhozeni 2. (nad hlavni) az nejvrchnejsi diagonaly
        mat_2 = np.tril(m=np.ones(shape=quant_relations.shape, dtype=int), k=-2) # vyhozeni -2. (pod hlavni) az nejspodnejsi diagonaly
        #BinTestMatrix(tril(ones(k,k),-2)==1)=0; % vyhozeni -2. (pod hlavni) az nejspodnejsi diagonaly
        #BinTestMatrix(tril(ones(k,k),1)==0)=0; % vyhozeni 2. (nad hlavni) az nejvrchnejsi diagonaly
        quant_adjac = mat_1 - mat_2
        quant_relations = quant_relations * quant_adjac
        # => zbydou dvojice slozene jen ze sousedicich kvantilu (nesousedici nema smysl porovnavat)
    
        ind_max = np.unravel_index(np.argmax(quant_relations, axis=None), quant_relations.shape) # dvojice kvantilu s nejvetsi podobnosti    
        pMax = quant_relations.iloc[ind_max]
        if pMax>pConf: # pokud jsou si nektere kvantily prilis podobne (slouci se)
            
            q1_label = quant_info["q_label"].iloc[ind_max[0]]
            q2_label = quant_info["q_label"].iloc[ind_max[1]]
            q_label_new = max(q1_label, q2_label)
            q_attr = {q1_label: q_label_new, q2_label: q_label_new} # ktery hodnoty nahradit cim
    
            # slouceni v in-samplu i out-samplu = prepsani hodnot/labelu danych kvantilu (tou vyssi z hodnot/labelu)
            predictor_cat_in = predictor_cat_in.replace(q_attr) 
            predictor_cat_out = predictor_cat_out.replace(q_attr) # kdyby nektere labely byly jen v in a ne v out, tak neni chyba (jen se nenahradi)
            # klasicka detekce shody nefunguje (zaokrouhleni), ale replace dictem ano
            #predictor_cat_in.loc[predictor_cat_in==ind_max[0]] = q_label_new # neprobehne to 
            #predictor_cat_in.loc[predictor_cat_in==ind_max[1]] = q_label_new # neprobehne to 
            #len(predictor_cat_in.loc[predictor_cat_in==ind_max[0]]) # 0 => asi doslo k zaokrouhleni => jednodussi labely zvlast
            
            # kdyz uz tam je jen jedina kategorie (poslucovalo se tak), nevolat znova quantiles_success_test
            # ve proportions_chisquare_allpairs ==> ValueError: zero-size array to reduction operation maximum which has no identity
            n_categ = len(pd.unique(predictor_cat_in))
    
    # vystupni transformovana data
    predictor_cat_all = pd.concat(objs=[predictor_cat_in, predictor_cat_out], axis="index")
    quant_info_in, _ = quantile_stats_categoric(predictor_cat_in, target_bin_in)
    quant_info_out, _ = quantile_stats_categoric(predictor_cat_out, target_bin_out)
    quant_info_all, _ = quantile_stats_categoric(predictor_cat_all, target_bin)
    
    return predictor_cat_all, quant_info_in, quant_info_out, quant_info_all



def category_merging_transformation(predictor_cat: pd.Series, target_bin: pd.Series, pConf: float, outsample_start: str):
    """
    Binning_All_Out_v2 - Matlab
    
    Wrapper for category_merging_predictor output usage - replacing of predictor values with merged quantile values

    Parameters
    ----------
    predictor_cat : pd.Series
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    pConf : float
        maximal similarity of categories for not joining them - p-value from test
        
    outsample_start : str
        start date of out-sample partition of data

    Returns
    -------
    predictor_cat_merged : pd.Series
        predictor values with merged quantiles

    """

    # zbavit se ostatnich "informacnich" vystupu pro .apply
    predictor_cat_merged, _, _, _ = category_merging_predictor(predictor_cat, target_bin, pConf, outsample_start)
    
    return predictor_cat_merged



def univar_analysis_stats_cat(predictor_cat: pd.Series, target_bin: pd.Series):
    """
    Univariate_Analysis_v3 - Matlab
    
    Compute performance statistics of one categoric predictor - from analysis of impact of predictor categories on target variable.
    Similar to univar_analysis_stats() but for categoric data (no regression etc.)
    
    Parameters
    ----------
    predictor : pd.Series
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    Returns
    -------        
    IV_predictor : float
        information value of the predictor (cumulative woe of all quantiles)
        
    chi2pval : float
        p-value of the chi-squared test (independence of the successfulness in the categories/quantiles)

    """
    
    min_pos, min_nonpos = 1, 1 # min. variabilita pro woeizaci: pocet klad. a pocet neklad. returnu v kvantilu
    
    ## Category_Woeization - woeizace na kvantilovanych prediktorech
    woeized_factor, quantiles_info, IV_predictor = category_woeization_predictor(predictor_cat, target_bin, min_pos, min_nonpos) 
    # woeized_factor - obsahuje i kategoricky data (XPD) i woeizovany (XW)
    
    # DefaultPD - z kateg. dat vyextrahuje kategorie a jejich hit rate -> neni potreba: uz je vystupem category_woeization_predictor()
    #quant_info, quantiled_factor = quantile_stats_categoric(predictor_cat, target_bin)

    # QuickGini - vystupy z toho predane dale (AR) jako vystup Univariate_Analysis_v3 nepouzity v "hlavnim" skriptu
    # .. vstupy jsou: bin. target a kvantilovany ("target_qmean") prediktor

    ## ChiSquaredTest - test vlivu kategorizace na uspesnost v kvantilu
    chi2pval = chi2_test(woeized_factor["target_qmean"], target_bin)
    
    # ulozit statistiky do vystupu
    return IV_predictor, chi2pval



def category_woeization_predictor_insample(predictor_cat: pd.Series, target_bin: pd.Series, min_pos: int, min_nonpos: int, outsample_start: str):
    """
    Category_Woeization_Out - Matlab
    
    Replacement of predictor quantile values by quantile WoE computed on in-sample data partition
    Similar to category_woeization_predictor, but trained only on in-sample 

    Parameters
    ----------
    predictor_cat : pd.Series
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    min_pos : int
        minimal number of positive values/returns in quantile
        
    min_nonpos : int
        minimal number of negative/nonpositive values/returns in quantile
        
    outsample_start : str
        start date of out-sample partition of data

    Returns
    -------
    woeized_factor_all : pd.Series
        whole sample of predictor values replaced with quantile WoE

    """

    missing_cat_val = 0 # defaultni hodnota pro kategorie, ktere nebyly v in-samplu

    # woeizace - "uceni" na in-sapmlu, pouziti na full-samplu (u out-sample kv. neexistujicich v in-samplu dat default hodnotu)    
    in_predictor_cat = predictor_cat.loc[slice(None), :outsample_start].copy() # v in-sample nebude posledni datum
    in_target_bin = target_bin.loc[slice(None), :outsample_start].copy() 
    
    in_woeized_factor, in_quantiles_info, in_IV_predictor = category_woeization_predictor(in_predictor_cat, in_target_bin, min_pos, min_nonpos) 
    # woeized_factor_in - obsahuje i kategoricky data (XPD) i woeizovany (XW)
    
    # nahradit v datech hodnoty kvantilu prislusnym kvantilovym WoE
    q_woe = in_quantiles_info[["q_label", "WoE"]].set_index("q_label")["WoE"].to_dict()
    # pridat do q_woe dictu pripadne klice z out-samplu ktere nejsou v in-samplu
    missing_keys = list( set(pd.unique(predictor_cat)) - set(q_woe.keys()) ) 
    for missing_key in missing_keys:
        q_woe[missing_key] = missing_cat_val
    
    woeized_factor_all = predictor_cat.replace(q_woe) # nahrazeni vahami Woe v celych datech
    
    # woeized_factor_in, IV_predictor_in, quantiles_info_in nevracet - nejsou dale pouzite 
    return woeized_factor_all



def quick_gini(target_bin: pd.Series, predictor_num: pd.Series, target_class: int):
    """
    QuickGini - Matlab
    
    Calculation of Gini Accuracy Ratio

    Parameters
    ----------
    target_bin : pd.Series
        binarized target 

    predictor_num : pd.Series
        one categoric/quantiled feature
        
    target_class : int
        indicator of the predicted (desired) class 

    Returns
    -------
    AR_score : float (or np.nan)
        predictor performance (prevalence of correct classification above wrong classification)

    """
    if len(target_bin)==0:
        AR_score = np.nan
    else:
    
        if min(target_bin)==max(target_bin): # neni zadna variabilita
            AR_score = np.nan
        else:
            YF_Good = predictor_num.loc[target_bin==target_class] # hodnoty faktoru v chtenych/uspesnych dnech/obdobich
            YF_Bad = predictor_num.loc[target_bin!=target_class] # hodnoty faktoru v nechtenych/uspesnych dnech/obdobich
            n_good = len(YF_Good) # pocet uspesnych obdobi
            n_bad = len(YF_Bad) # pocet neuspesnych obdobi
            n_pairs = n_good*n_bad # pocet dvojic "kazdy s kazdym" (neuspesna vs. uspesna obdobi)
            
            # 2 stejne dlouhe (n_good*n_bad) pole - vsechny dvojice "uspesne-neuspesne" obdobi
            YF_Pairs_1 = np.repeat(a=YF_Good, repeats=n_bad).to_numpy() # namnozeni
            YF_Pairs_2 = np.tile(A=YF_Bad, reps=n_good) # "dlazdicove" nakopirovani
            YF_Pairs = np.column_stack(tup=(YF_Pairs_1, YF_Pairs_2)) # vsechny dvojice "uspesne-neuspesne" obdobi
            # -- 1. sloupec - prvky v YF_Good n_bad-krat namnozene (1 1 1 2 2 2 3 3 3 atd.)
            # -- 2. sloupec - cele YF_Bad n_good-krat nakopirovane (1 2 3 1 2 3 1 2 3 atd.)
            
            #n_correct = sum( YF_Pairs[:,0]>YF_Pairs[:,1] ) # strasne dlouho se pocita ta suma
            YF_correct = YF_Pairs[:,0]>YF_Pairs[:,1] # toto je hned
            #n_correct = sum( YF_correct ) # strasne dlouho se pocita
            YF_correct_trues = YF_correct[YF_correct] # hned spocitane
            n_correct = len(YF_correct_trues) # hned spocitane
            
            # kolikrat je "to spravne" = faktor (vahy WoE n. jine) u uspesnych (sl. 1) je vetsi nez u neuspesnych (sl. 2)
            YF_wrong = YF_Pairs[:,0]<YF_Pairs[:,1]
            n_wrong = len( YF_wrong[YF_wrong] ) # kolikrat je "to spatne"
            
            p_correct = n_correct/n_pairs # v jakem podilu dvojic bylo "spravne zarazeni" do kvantilu
            p_wrong = n_wrong/n_pairs # -"- spatne zarazeni
            AR_score = p_correct-p_wrong # jak moc prevazuje mira spravneho urceni nad mirou spatneho
    
    return AR_score



def remove_correlated_cat(predictors_stacked: pd.DataFrame, target_bin: pd.Series, target_class: int, maxCorr: float):
    """
    Correlation_Remove - Matlab
    
    Filter out the categoric predictors simultaneously having high correlation with other predictors and lower performance (Accur. Ratio)
    Similar to remove_correlated() but for categorical variables
    
    Parameters
    ----------
    predictors_stacked : pd.DataFrame
        evolution of all features for all assets in portfolio in long/stacked form (nDays*nAssets rows x nFeatures columns)
    
    target_bin : pd.Series
        binarized target - for Gini Accuracy Ratio calculation

    target_class : int
        indicator of the predicted (desired) class - for Gini Accuracy Ratio calculation
        
    maxCorr : float
        maximum tolerated correlation among predictors

    Returns
    -------
    predictors_out : pd.DataFrame
        filtered predictors (in long/stacked form)
        
    vars_keep : list
        names of kept predictors
        
    predictors_corr : pd.DataFrame
        corrected correlation matrix of predictors (high correlations of underperforming predictors are set to 0)

    """
    # QuickGini skore pro vsechny features
    # M. Ficura: woe (in our implementation - from clients scoring) is working in opposite way than final score
    # .. (the lower woe is the higher probability of the target occurence)
    predictors_woe_neg = -predictors_stacked # negative woeized predictors: [AR] = QuickGini(Y,-XW(:,j),1); 
    predictors_perf = predictors_woe_neg.apply(lambda col: quick_gini(target_bin, col, target_class), axis="index", result_type="expand")

    predictors_corr = predictors_stacked.corr(method="pearson") # NaNy jsou ignorovane
    
    # "vyhozeni" vysokych korelaci sama se sebou - "label-based" diagonal filling 
    #idx_diag = predictors_corr.index.intersection(other=predictors_corr.columns) # klice existujici v indexu i columns
    #diag_vals = pd.Series(0, index=pd.MultiIndex.from_arrays(arrays=[idx_diag, idx_diag])) # hodnoty a jejich souradnice
    #predictors_corr.stack(dropna=False).update(diag_vals) # Ficura ale nechava vysoke kor. sama se sebou... ????
    # pro kazdou promennou: z promennych "sama sebe" a prip. dalsich s vysokyma korelacema vybere tu s nejvyssi performance a ponecha ji
    # u ostatnich s vysokyma korelacema je odstrani
    
    high_corr = abs(predictors_corr)>maxCorr
    
    vars_keep = list(predictors_stacked.columns) # na zacatku vsechny ponechat
    
    if high_corr.sum().sum()>len(high_corr): # kdyz se vyskytla vysoka korelace i mimo diagonalu
        #corr_vars = high_corr.index[high_corr.any()] # promenne s necim vysoce korelujici v pripade nul na hlavni diagonale
        corr_vars = predictors_stacked.columns # vzdy je vysoka korelace min. sama se sebou
        # pro kazdou promennou s necim vysoce korelujici nechat jen tu s nejvetsi p-hodnotou
        for corr_var in corr_vars:
            corr_vars_perfs = predictors_perf.loc[high_corr.loc[corr_var, :]] # "vykonnosti" promennych s necim vysoce korelujicich
            best_predictor = corr_vars_perfs.idxmax(axis=0) # prediktor s nejlepsim performance
            predictors_corr.loc[corr_var, best_predictor] = 0 # oznaceni jako nizke korelace => ponechat
            #predictors_corr.loc[best_predictor, corr_var] = 0 # zrcadlove -"-
    
        #high_corr_final = abs(predictors_corr)>maxCorr
        low_corr_final = abs(predictors_corr)<maxCorr
        vars_keep = low_corr_final.all(axis="index") # low korelace musi byt se vsema
        vars_keep = list(vars_keep.index[vars_keep]) # aby fce vracela vzdy stejny typ (list vs. index. vs. series)
    
    predictors_out = predictors_stacked.loc[:, vars_keep]

    return predictors_out, vars_keep, predictors_corr



def features_binning_filtering(predictors_stacked: pd.DataFrame, target_stacked: pd.Series, target_bin: pd.Series, outsample_start: str, 
                               nQ_woe: int, nQuant: int, pValSelect: float, pConf: float, pKeep: float, min_pos: int, min_nonpos: int, target_class: int, maxCorr_cat: float):
    """
    StockPicking_Model_Binary - Matlab
    
    Converts numeric features to categoric and filter out the insignifficant ones - wrapper for pipeline using:
        univar_analysis_stats, 
        factor_qmax_transformation, 
        category_merging_transformation, 
        univar_analysis_stats_cat, 
        category_woeization_predictor_insample
        remove_correlated_cat 

    Parameters
    ----------
    predictors_stacked : pd.DataFrame
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_stacked : pd.Series
        evolution of target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_bin : pd.Series
        evolution of binarized target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    outsample_start : str
        start date of out-sample partition of data
        
    nQ_woe : int
        number of quantiles for woeization
        
    nQuant: int
        number of quantiles the predictor will be splitted to
        
    pValSelect : float
        p-value for variable pre-selection according linreg and chi2 test
        
    pConf : float
        maximal similarity of categories for not joining them - p-value from test
        
    pKeep : float
        chi2 p-value used for the selection of the binned variables
        
    min_pos : int
        minimal number of positive values/returns in quantile
        
    min_nonpos : int
        minimal number of negative/nonpositive values/returns in quantile
        
    target_class : int
        target/predicted class in quick gini
        
    maxCorr_cat : float
        maximum allowed correlation between predictors

    Returns
    -------
    predictors_woe_filtered: pd.DataFrame
        whole sample of predictor values replaced with quantile WoE (calulated on in-sample), insignifficant predictors filtered out
    
    predictors_cat_filtered: pd.DataFrame
        predictor values transformed to categoric (with merged quantiles), insignifficant predictors filtered out

    """

    ### rozdeleni na in- a out-sample podle datumu
    target_in = target_stacked.loc[:, :outsample_start, :]
    target_bin_in = target_bin.loc[:, :outsample_start, :]
    predictors_in = predictors_stacked.loc[:, :outsample_start, :]
    
    ## filtrovani regresi a chisq testem - preselekce stat. vyznamnych faktoru
    #u_b, u_p, u_r2, u_IV_predictor, u_chi2pval = univar_analysis_stats(predictor_stacked, target_stacked, target_bin) # pro jednotlivy faktor
    # statistiky z jednorozmerne analyzy - pro vsechny faktory 
    stats_univar = predictors_in.apply(lambda col: univar_analysis_stats(col, target_in, target_bin_in, nQ_woe), axis="index", result_type="expand")
    stats_univar.index=["linreg_beta1", "linreg_pval", "linreg_r2adj", "woe_IV", "chi2_pval"]
    stats_univar = stats_univar.T # stejny format jako f-m perf. stats
    # kt. faktory splnuji aspon v 1 p-hodnote (vyznamny aspon 1 test)
    factors_selected = list(stats_univar.index[ (stats_univar.loc[:, ["linreg_pval", "chi2_pval"]]<pValSelect).any(axis="columns") ])
    # totez: mensi z ukazatelu p-val beta1 a p-val ChiSq => alespon jeden mensi nez hladina vyznamnosti
    #factors_selected = list(stats_univar.index[ stats_univar.loc[:, ["linreg_pval", "chi2_pval"]].min(axis="columns")<pValSelect ])
    if len(factors_selected)==0: # nema smysl uz nic dal pocitat
        print("Categoric features, univar. analysis - all features discarded")
        return
    predictors_univar = predictors_stacked.loc[:, factors_selected] # XS v StockPicking_Model_Binary
    
    ## prepocitani kvantilu: kvantily se "uci"/spocitaji jen na in-sample datech a to se pak pouzije na olabelovani celych dat
    # nahradit kvantilovymi maximy faktoru
    predictors_qmax = predictors_univar.apply(lambda col: factor_qmax_transformation(col, target_stacked, nQuant, outsample_start), axis="index", result_type="expand")
    # = XQ2 v StockPicking_Model_Binary
    
    ## binning - Binning_All_Out_v2 - zatim neimplementovana varianta 2: Binning_PD_Ordinal_Monotonic_Out
    #BinType=2; % Type of binning, 1=Binomial, 2=Binomial & Monotonic
    #% Type = 1=Quantiles ordinal binning, 2=Continuous variable tree binning
    #nMin=10 # Minimum number of observations in a bin (for CS tree)
    # Categories_Binomial_Test - ma tam byt 1-pval a ne jen pval ???????
    predictors_cat_merged = predictors_qmax.apply(lambda col: category_merging_transformation(col, target_bin, pConf, outsample_start), axis="index", result_type="expand")
    """
    /opt/anaconda3/lib/python3.8/site-packages/statsmodels/stats/base.py:108: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use 
    `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      pvals_mat[lzip(*self.all_pairs)] = self.pval_corrected()
      
    # je to po (v quantiles_success_test() uvnit category_merging_predictor()):
        pval_mat = result_prop.pval_table() # jen horni trojuhelnik, 
    # .. asi s tim nejde nic delat - jak jinak uchopit prvek AllPairsResults
    """
    # pro jistotu seradit aby nebyl problem s datumovym vyberem - v transfornacich asi nezustane vzdy zachovane razeni
    #UnsortedIndexError: 'MultiIndex slicing requires the index to be lexsorted: slicing on levels [1], lexsort depth 0'
    predictors_cat_merged = predictors_cat_merged.sort_index(level=[0, 1], ascending=[True, True])
    # vynechat kontrolu binningu - Ficura to pak dale nevyuziva
    # .. pocet kv. v in-samplu = pocet kv. v out-samplu => u kterych promennych to slouceni "skoncilo" stejne v in-samplu a out-samplu
    # Remove variables with insignificant binning = nechat jen ty, kde zustalo 2 a vice binu
    vars_keep = predictors_cat_merged.nunique(axis="index")>1
    if len(vars_keep)==0: # nema smysl uz nic dal pocitat
        print("Categoric features, category merging - all features discarded")
        return
    predictors = predictors_cat_merged.loc[:, vars_keep] # XB2 v StockPicking_Model_Binary
    
    ## odfiltrovani kategorickych promennych, ktere se po slouceni kvantilu staly nevyznamnymi - chisq testem
    #uc_IV_predictor, uc_chi2pval = univar_analysis_stats_cat(predictor_cat, target_bin) # pro 1 faktor
    stats_univar_cat = predictors.apply(lambda col: univar_analysis_stats_cat(col, target_bin), axis="index", result_type="expand")
    stats_univar_cat.index=["woe_IV", "chi2_pval"]
    stats_univar_cat = stats_univar_cat.T # stejny format jako f-m perf. stats
    factors_selected = list(stats_univar_cat.index[ (stats_univar_cat.loc[:, "chi2_pval"]<pKeep) ])
    if len(factors_selected)==0: # nema smysl uz nic dal pocitat
        print("Categoric features, categoric univar. analysis - all features discarded")   
        return
    predictors_cat = predictors.loc[:, factors_selected] # XB3 v StockPicking_Model_Binary
    
    ## WoEizace na in-samplu - kategoricke hodnoty faktoru nahrazene vahami WoE tech kategorii/kvantilu
    #predictors_cat = predictors_cat.sort_index(level=[0, 1], ascending=[True, True]) # pro jistotu seradit aby nebyl problem s datumovym vyberem 
    predictors_cat_woe = predictors_cat.apply(lambda col: category_woeization_predictor_insample(col, target_bin, min_pos, min_nonpos, outsample_start), axis="index", result_type="expand")
    # odfiltrovani pripadne konstatnich prediktoru vzniklych woeizaci
    nonconst_features = predictors_cat_woe.min(axis="index") != predictors_cat_woe.max(axis="index") # kde jsou vsechny hodnoty stejne
    predictors_cat_woe = predictors_cat_woe.loc[:, nonconst_features] # = XW v StockPicking_Model_Binary
    
    # kdyz je delsi predictors_* nez target_bin => chyba!!!!!
    ###IndexingError: Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).
    # => pro quick_gini() musi byt predictors_* i target stejne dlouhe -> upravit pred vstupem do remove_correlated_cat() pouzivajici quick_gini()
    tmp_data = pd.concat(objs=[predictors_cat_woe, target_bin], axis="columns").dropna(how="all", subset=[target_bin.name], axis="index")
    tmp_predictors_cat_woe = tmp_data.drop(labels=target_bin.name, axis="columns") # "osekane" predictors_cat_woe
    
    ## odfiltrovani prediktoru silne korelovanych na in-samplu
    predictors_woe_filtered, lowcorr_vars_cat, corr_predictors_cat = remove_correlated_cat(tmp_predictors_cat_woe, target_bin, target_class, maxCorr_cat)
    # predictors_woe_filtered: vyfiltrovane ty "zkracene" prediktory -> vyfiltrovat ty puvodni, co jsou delsi nez target
    predictors_woe_filtered = predictors_cat_woe.loc[:, predictors_woe_filtered.columns]
    
    # = XWC resp. XW prepsane hodnotami XWC v StockPicking_Model_Binary
    predictors_cat_filtered = predictors.loc[:, predictors_woe_filtered.columns] # XB5 v StockPicking_Model_Binary
    
    return predictors_woe_filtered, predictors_cat_filtered


# .. QuickGini pak pripadne doplnit do performance funkci i kdyz se tento jejich vystup zatim nikde nepouzival


"""
# pokus o vektorizaci vypoctu pri pouziti each_observation_transformation() ve factor_mimicking_returns()
vecF = np.vectorize(each_observation_transformation) 
tmp = pd.DataFrame(vecF(factor_all, target_all, 3))          # apply the function to two data frames

#df_results = factor_all.combine(other=target_all, func=lambda x,y: each_observation_transformation(x, y, nQuant) )
factor_all.combine(other=target_all, func=each_observation_transformation, nQuant=3 )
"""