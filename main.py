# -*- coding: utf-8 -*-

#%%
#from argparse import ArgumentParser
import data_computer as dc
import data_storage as ds
import feature_selection as fs
import pandas as pd
import numpy as np
import time
import datetime as dt
import warnings
import os

#parser = ArgumentParser()
#parser.add_argument("-f", "--rebalance-frequency", dest="freq_rebalanc", help="rebalance frequency", choices=['daily', 'monthly', 'quarterly', 'annual'], default="monthly")
#parser.add_argument("-i", "--stock-index", dest="stock_index", help="stock market index", choices=['SP500', 'Russel3000'])
#args = parser.parse_args()
from model_performance import backtest_stockpicker
from model_diagnostics import print_metrics, plot_learning_curve

###############################################################################
#
#                    META-PARAMETERS AND CONFIGURATION
#
###############################################################################
start_time = time.time() # measuring elapsed time
warnings.filterwarnings("ignore")

LITE_version = True # malo tickeru, malo prediktoru -> rychly vypocet (nekvalitni vysledky)

# Specify Stocks Universe 
tickers = ds.get_sp100()
SAVED_fmr_dir = "sp100" # slozka pro ulozeni napocitanych factor mimicking portfolio returns
if LITE_version:
    print("Using test universe - 7 tickers")
    tickers = ["AAPL", "FB", "AMZN", "MSFT", "IBM", "F", "CVX"] # TEST
    SAVED_fmr_dir = "test7"
tickers.sort() # ['AAPL', 'AMZN', 'CVX', 'F', 'FB', 'IBM', 'MSFT']

data_start = "1989-12-31" # zacatek historie OHLCV dat z YF

# Specify Investing Horizons
horizons = [5, 10, 22, 22*2, 22*3, 126, 252, 252*2, 252*3] # stat., momentum, cur. vol, ar logprice
horizons_technical = [2, 3, 4, 5, 10, 22, 22*2, 22*3, 126, 252, 252*2, 252*3] # tech., vol. tech.
horizons_smooth_5 = [10, 22, 22*2, 22*3, 126, 252, 252*2, 252*3] # horizons vetsi nez smoothing
horizons_smooth_22 = [22*2, 22*3, 126, 252, 252*2, 252*3] # horizons vetsi nez smoothing
horizons_suv = [22, 44, 66, 126, 252] # std. unexpl. vol.
horizons_w = [2, 3, 4, 8, 12, 26, 52, 104, 156] # weekly technical, obv, volume technical
horizons_suv_w = [8, 12, 26, 52, 104, 156] # weekly std. unexpl. vol.
horizons_ar_w = [4, 8, 12, 26, 52, 104, 156] # weekly ar
if LITE_version:
    horizons = horizons_technical = horizons_smooth_5 = horizons_smooth_5 = horizons_suv = horizons_w = horizons_suv_w = horizons_ar_w = [15, 22, 66] # TEST
horizL = [126, 126, 126, 252, 252, 252, 504, 504, 504]
horizS = [10,  22,  44,  10,  22,  44,  10,  22,  44] 
horiz_pairs_mom = list(zip(horizS, horizL)) # momentum 
horizL = [10, 22, 44, 66, 44, 66, 126, 252, 126, 252, 504, 756, 504, 756]
horizS = [5,  5,  5,  5,  22, 22, 22,  22,  66,  66,  66,  66,  252, 252] 
horiz_pairs_currvol = list(zip(horizS, horizL)) # curr. vol.
if LITE_version:
    horiz_pairs_mom = horiz_pairs_currvol = [ [15, 22], [15, 66], [22, 66] ] # TEST

# Frequency of portfolio re-balancing - iFrequency
freq_rebalanc = "monthly" # daily, weekly, monthly, quarterly, semi-annual, annual
#freq_rebalanc = args.freq_rebalanc 
# zatim podporovano jen daily, monthly, quarterly, annual
# Number of periods in a year - PeriodsYear
n_periods_year = {"daily": 252,
                  #"weekly": 52, # patky
                  "monthly": 12, 
                  "quarterly": 4,
                  #"semi-annual": 2, 
                  "annual": 1}

# basic meta-parameters - for calculations prior to all models
nQuant = 3
kYear = n_periods_year[freq_rebalanc] # number of periods/observations in a year
#pVal=0.101 # aby neco vyslo, ale nejlepe napr. 0.01, Ficura ma 0.001 
pVal=0.9901 # aby neco vyslo, ale nejlepe napr. 0.01, Ficura ma 0.001 
maxCorr = 0.6 # 0.0015 - aby vysla nejaka korelace vysoka, ale jinak napr. 0.6

outsample_start = "2008-01-01" # 1990-2007 dle Ficury pro trenink
#outsample_start = df.index[ int(np.ceil(len(df) * .2)) ].strftime("%Y-%m-%d") # datum dle pct
# jinak lze i rozdelit train_test_splitem dle pct a pak vyextrahovat datum zacatku out-samplu

# meta-parameters - specific for binary models with categoric data
nQ_woe = 10 # decily - pro woeizaci
pValSelect=0.305 # p-value for variable pre-selection according linreg and chi2 test, aby neco zustalo, jinak 0.05
pConf=0.405 # p-value used for binning, aby neco zustalo, jinak 0.05
pKeep=0.9905 # p-value used for the selection of the binned variables, aby neco zustalo, jinak 0.05
min_pos, min_nonpos = 1, 1 # min. variabilita pro woeizaci: pocet klad. a pocet neklad. returnu v kvantilu
maxCorr_cat = 0.10015 # aby vysla nejaka korelace vysoka, ale jinak napr. 0.6
target_class = 1 # iclass

# other configuration
#feat_start = 500; feat_stop = 700 # rozsah dat - neco malo na pokusy at to nepocita moc dlouho
#feat_start = 0; feat_stop = 999999 # vsechno: len(all_features) - i pro pripad TEST_data
max_pct_missing = .999 # maximalni podil chybejicich hodnot pro ponechani prediktoru 
nan_val_replace_predictor = 0 # nahrazeni NaNu v prediktorech dle typu transformace returnu (0.5 u Rank of returns a Quantile/Exponential/Log-Normal transformation jinak 0)
fm_verbose = True # prubezny vypis zpracovani featur ve all_factors_mimicking_returns()

# model types used for training
model_type = "linreg-ridge"
models_categoric = ["elnet-logreg"]

SAVED_data_dir = "../SP-data/" # slozka pro ukladani a cteni z vysledku mezivypoctu 
SAVED_factors_dir = os.path.join(SAVED_data_dir, "factors_daily") # zakl slozka pro ukladani a cteni dat prediktoru 
#%%
###############################################################################
#
#                               DATA PREPARATION
#
###############################################################################

### data harvesting
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Getting OHLCV data...")
df = ds.get_ohlcv_data(tickers, data_start, SAVED_data_dir)

### All Targets - "raw"
# HM_Ret=Difr(log(AA_AdjClose(IndTT,:))); - Ficura krome volume nedela zadny agregace, jen vybere prislusne hodnoty dle frekvence
# resampling jen pro target/return - musej se resamplovat ceny jeste pred vypoctem returnu (ale ostani features jsou na daily)
# resamplovat data na daily a nove vznikle dny/NaNy (svatky, vikendy atd.) vyplnit -> vybrat dny podle Ficury
# bez resamplovani a vyplneni by nektera pozorovani chybela (obchodni den vzdy neni na konci mesice apod.)
# kdyz neni konec obdobi, pouzit "castecna" data pro vypocet posledniho udaje
# ... radsi nez spolehat na pandas business- veci (business month end frequency atd.)
# weekly -> patek, monthly/quarterly/semi-anual/annual -> posledni den toho obdobi
if freq_rebalanc=="daily":
    df_resampled = df
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
all_target = dc.compute_return(df_resampled) # RuntimeWarning: invalid value encountered in true_divide 
all_target_scaled = dc.cross_sectional_standardize(to_scaling=all_target) # stand. target - az pro pouziti v predikcnim modelu

if (not os.path.exists(SAVED_data_dir)):  # create dir if does not exist
    os.makedirs(SAVED_data_dir)
    print("Directory for using precomputed data created: ", os.path.abspath(SAVED_data_dir))
        
### All Features - pro pocitani f-m returns filtraci pred vstupem do modelu v unstacked podobe, stackne se to az po filtraci
# pocita se na nejpodrobnejsich datech - resampling az u features spocitanych na daily
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Processing of statistical moments factors...")
#statistical_moments_test = dc.statistical_moments(df, horizons) # stejne warningy jako u currency_volume()
statistical_moments_test = ds.get_statistical_moments(df, horizons, SAVED_factors_dir) # stejne warningy jako u currency_volume()
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Calculation of momentum factors...")
momentum = dc.momentum(df, horizons, horiz_pairs_mom) # stejne warningy jako u currency_volume()
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Calculation of currency volume factors...")
currency_volume = dc.currency_volume(df, horizons, horiz_pairs_currvol) # invalid value encountered in true_divide + Degrees of freedom <= 0 for slice.
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Calculation of volume factors...")
cumul_volume = dc.compute_volume(df)
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Processing of price technical factors...")
#technical_factors = dc.technical_factors(df, horizons_technical) # stejne warningy jako u currency_volume()
technical_factors = ds.get_price_technical(df, horizons_technical, SAVED_factors_dir) # stejne warningy jako u currency_volume()
#% 7. COMPUTE OBV FACTORS
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Processing of volume technical factors...")
#volume_technical_factors = dc.volume_technical_factors(df, horizons_technical) # stejne warningy jako u currency_volume()
volume_technical_factors = ds.get_volume_technical(df, horizons_technical, SAVED_factors_dir) # stejne warningy jako u currency_volume()
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Calculation of smoothed volume 5 technical factors...")
smooth_vol_technical_factors_5 = dc.smoothed_volume_technical_factors(df, 5, horizons_smooth_5) # stejne warningy jako u currency_volume()
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Calculation of smoothed volume 22 technical factors...")
smooth_vol_technical_factors_22 = dc.smoothed_volume_technical_factors(df, 22, horizons_smooth_22) # stejne warningy jako u currency_volume()
#% 11. COMPUTE DIVIDEND PAYOUT YIELD (DPY) FACTORS
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Processing of SUV factors...")
#suv_factors = dc.standardized_unexplained_volume(df, horizons_suv)
suv_factors = ds.get_standardized_unexplained_volume(df, horizons_suv, SAVED_factors_dir)

if not LITE_version: # v jednoduche verzi vynechat narocny vypocet AR faktoru
    print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Processing of AR factors...")
    ar1_logprice = ds.get_ar1_logprice(df, horizons, SAVED_factors_dir) # strasne dlouho trva prvni vypocet !!!

# + technical, obv, vol. tech., suv, ar jsou jeste ve weeks variante (sk. 14-18) - vyznamne jen u suv a ar
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Calculation of SUV factors weeks...")
#warnings.filterwarnings("default")
#suv_factors_w = dc.standardized_unexplained_volume_weeks(df, horizons_suv_w)
suv_factors_w = ds.get_standardized_unexplained_volume_weeks(df, horizons_suv_w, SAVED_factors_dir) # dopocet nekonzistentni s celym vypoctem
# NEKONZISTENCE u F agg resid 15 a coef 15 - u pocitaneho jsou vsude nany, pri ukladani/dopocitani jsou na konci (u chybejicich dni) cisla
# v dc.standardized_unexplained_volume() pri odchytu vyjimky se cely pocitany sloupec vyplni NaNy, v dopoctu (kratsi data) to nastat nemusi
#########################################################################


if not LITE_version: # v jednoduche verzi vynechat narocny vypocet AR weeks faktoru
    print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Processing of AR factors weeks...")
    ar1_logprice_w = ds.get_ar1_logprice_weeks(df, horizons_ar_w, SAVED_factors_dir) # strasne dlouho trva vypocet !!!

if not LITE_version:
    print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Calculation of fundamental factors...")
    fundamental = dc.get_fundamentals(tickers, horizons, SAVED_data_dir) # vrati pripadne prazdny df pri chybe
#%% Concat All Features
if LITE_version:
    # VYNECHANA VETSINA PREDIKTORU, ABY F-M RETURNS NETRVALY DLOUHO
    all_features = pd.concat(objs=[statistical_moments_test, suv_factors ], axis="columns").sort_index(axis="columns") # TEST
else:
    all_features = pd.concat(objs=[
        statistical_moments_test,
        momentum,
        currency_volume,
        cumul_volume,
        technical_factors,
        #% 7. COMPUTE OBV FACTORS - vynechane (nevyznamne)
        volume_technical_factors,
        smooth_vol_technical_factors_5,
        smooth_vol_technical_factors_22,
        #% 11. COMPUTE DIVIDEND PAYOUT YIELD (DPY) FACTORS - vynechane (nevyznamne)
        suv_factors, 
        ar1_logprice, #% 13. AR FACTORS - velmi narocny vypocet
        #% 14. COMPUTE TECHNICAL FACTORS - WEEKS - vynechane (nevyznamne)
        #% 15. COMPUTE OBV FACTORS - WEEKS - vynechane (nevyznamne)
        #% 16. COMPUTE VOLUME INDICATORS - WEEKS - vynechane (nevyznamne)
        suv_factors_w,
        ar1_logprice_w, #% 18. AR FACTORS - velmi narocny vypocet
        fundamental # prazdny df pokud nastal nejaky problem pri zpracovani
    ], axis="columns").sort_index(axis="columns")


# Standardize cross-sectionally all features
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Standardization of factors...")
all_features_scaled = dc.make_standardization(features_to_scale=all_features)

"""
# tohle je asi blbe, resp. neni to presne podle zpusobu Ficury
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
# https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
resampling_rules = {"daily": "1D",
                  #"weekly": "1W", 
                  "monthly": "1M", 
                  #"quarterly": "Q", #Q nebo 3M?
                  #"semi-annual": "6M", 
                  "annual": "A"}
resample_rule = resampling_rules[freq_rebalanc] 

def resample_yf_data(data, target_freq):
    data.columns = data.columns.droplevel(level=0) # nechat jen level s OHLCV atd. (bez tickeru)
    
    data_resampled = data.resample(rule=target_freq, axis="index").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", 
                                          "Volume": sum, "Dividends": sum, "Stock Splits": "mean"}) 
                                            # u stock splits asi zadna agregace nedava smysl
    return data_resampled

# kdyz je denni rebalance - neresamplovat (vznikly by vikendove NaNy)
if freq_rebalanc!="daily":
    df = df.groupby(axis="columns", level=0).apply(lambda x: resample_yf_data(x, resample_rule))
#df_resampled = df.groupby(axis="columns", level=0).apply(lambda x: resample_yf_data(x, "1M"))
"""

print("Resampling of factors...")
if freq_rebalanc=="daily":
    all_features_resampled = all_features
    all_features_scaled_resampled = all_features_scaled
    
if freq_rebalanc=="monthly":
    all_features_resampled = all_features.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index")
    idx_resampled = all_features_resampled.index.is_month_end
    idx_resampled[-1] = True # posledni neukonceny i ukonceny mesic bude zahrnut vzdy
    all_features_resampled = all_features_resampled.loc[idx_resampled] 
    
    # to stejne se standardizovanyma features - pouzite az pro odhad modelu, ne pro preselekci
    all_features_scaled_resampled = all_features_scaled.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index")
    idx_resampled = all_features_scaled_resampled.index.is_month_end
    idx_resampled[-1] = True 
    all_features_scaled_resampled = all_features_scaled_resampled.loc[idx_resampled] 
    
if freq_rebalanc=="quarterly":
    all_features_resampled = all_features.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index")
    idx_resampled = all_features_resampled.index.is_quarter_end
    idx_resampled[-1] = True # posledni neukonceny i ukonceny kvartal bude zahrnut vzdy
    all_features_resampled = all_features_resampled.loc[idx_resampled] 
    
    # to stejne se standardizovanyma features - pouzite az pro odhad modelu, ne pro preselekci
    all_features_scaled_resampled = all_features_scaled.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index")
    idx_resampled = all_features_scaled_resampled.index.is_quarter_end
    idx_resampled[-1] = True 
    all_features_scaled_resampled = all_features_scaled_resampled.loc[idx_resampled] 
    
if freq_rebalanc=="annual":
    all_features_resampled = all_features.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index")
    idx_resampled = all_features_resampled.index.is_year_end
    idx_resampled[-1] = True # posledni neukonceny i ukonceny rok bude zahrnut vzdy
    all_features_resampled = all_features_resampled.loc[idx_resampled] 
    
    # to stejne se standardizovanyma features - pouzite az pro odhad modelu, ne pro preselekci
    all_features_scaled_resampled = all_features_scaled.resample(rule="1D", axis="index").agg("first").fillna(method="ffill", axis="index")
    idx_resampled = all_features_scaled_resampled.index.is_year_end
    idx_resampled[-1] = True 
    all_features_scaled_resampled = all_features_scaled_resampled.loc[idx_resampled] 
    

### data validity check
# pro jistotu kontrola pojmenovani dat
if all_features_resampled.columns.duplicated().sum()>0:
    print("Duplicate names in features ", all_features_resampled.columns[all_features_resampled.columns.duplicated()] )
    all_features_resampled = all_features_resampled.loc[:, ~all_features_resampled.columns.duplicated()] # odstraneni duplikatu

# pripadny podvyber na testovani - at netrva dlouho vypocet
features_all = all_features_resampled#.iloc[feat_start:feat_stop] 
target_all = all_target#.iloc[feat_start:feat_stop]

# kontrola - nejakou budem muset udelat nez to pustime do vypoctu
n_const = sum(features_all.min(axis="index") == features_all.max(axis="index")) # u kolika features jsou vsechny hodnoty stejne
print("Number of constant features: ", n_const)
features_missing_pct = (features_all.isna().sum(axis="index")/len(features_all)).sort_values(ascending=False) # kolik toho kde chybi
print("Percentages of missing values in features (top 50): ", features_missing_pct.iloc[:50])
# features, kde u nekterych tickeru chybi vse (nebo velmi hodne)
features_missings = features_all.columns[ (features_all.isna().sum(axis="index")/len(features_all))>max_pct_missing]
features_missings = set( features_missings.droplevel(level=0) ) # tyto features vyhodit, pripadne resit proc jsou tam samy nany
features_names = set( features_all.columns.droplevel(level=0) )
features_keep = list( features_names - features_missings)
# jen tak zkusit par featur kvuli rychlosti vypoctu
#features_keep = ["CurrencyVolume_15", "CurrencyVolumeAbsChange_15vs22", "CurrencyVolumeRelChange_15vs66", 
#                 "TRANGE", "ARRONOSC_15", "ARRONOSC_22", "ARRONOSC_66"]
features_all = features_all.loc[:, (slice(None), features_keep)].copy()

# zadany pocet kvantilu nema byt vetsi nez nejnizsi (v ramci dni) pocet (aktivnich) assetu
nQuant_max = ( len(tickers) - features_all.stack(level=1).isna().sum(axis="columns") ).min() 
if nQuant > nQuant_max:
    print("Number of desired quantiles is greater than maximum possible number of quantiles in data!!!")

### vyber na in-sample jen pro vypocet f-m returns
features_in = features_all.loc[:outsample_start, :]
target_in = target_all.loc[:outsample_start, :]

### factor mimicking returns pro vsechny faktory (pro vsechny assety) - nacist pripadne existujici a dopocitat zbytek
print(dt.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), "Factor-mimicking portfolio returns...")
path_fmret_port = os.path.join(SAVED_data_dir, "port_fm_returns", SAVED_fmr_dir, freq_rebalanc) # pro pouziti ve fci na ulozeni a dopocitani f-m-r
_, port_fmret_diff_cum, _, port_fmret_cum, port_fmcorr = ds.get_factor_mimicking_returns(features_in, target_in, nQuant, path_fmret_port)
#port_fmret_diff, port_fmret_diff_cum, port_fmret, port_fmret_cum, port_fmcorr 

#%%
### statistiky FM returnu pro nasledne filtrovani features
fm_perf_stats = fs.portfolio_perf_stats(port_fmret_diff_cum, port_fmret_cum, port_fmcorr, kYear)

# dale pouzivat kros-sekcne normalizovane (resamplovane) faktory i returny/targety

### preselekce dle splneni FMP p-hodnoty nebo Spearman kor. p-hodnoty
factors_selected = list(fm_perf_stats.index[ (fm_perf_stats.loc[:, ["T_pval", "lport_T_pval"]]>=(1-pVal)).any(axis="columns") ])
factors_data = all_features_scaled_resampled.loc[:, (slice(None), factors_selected)].copy()
if len(factors_selected)==0: # nema smysl uz nic dal pocitat
    print("F-m returns analysis - all features discarded")

# Apply directionality conditions = ty, co prosly + musi patrit do directional prediktoru
#IndFactors=IndFactors & ((FM_Dir==0) | (FM_Dir==FM_Stats_In(:,1))); 
# .. FM_All_Captions_YahooForecasts_Monthly_6Y, FM_All_Captions_MacroForecasts_Monthly_6Y
# Apply Bayesian selection conditions -- neni jeste vyjasneno

### presypat pod sebe faktory pro kazdy asset: StockPicking_ComputeSample - Matlab
# ve factors_data jsou i pozorovani, kt. jsou v predictors_stacked dropnuty kvuli stejne delce X a Y (nebyl k nim return po shiftnuti) 
factors_stacked = factors_data.stack(level=0).swaplevel(i=0, j=1, axis="index").sort_index(axis="index")
# prizpusobit target (returny)
target = all_target_scaled.copy()
target.columns = pd.MultiIndex.from_product(iterables=[ list(target.columns), ["Return"] ]) # pridani levelu
# .. shiftnout returny "nahoru" (vcerejsi hodnoty prediktoru k dnesni hodnote targetu)
target = target.shift(periods=-1, axis="index")  
# .. a zkratit faktory podle dostupnych returnu (nedostupne ret. se budou predikovat) a "sestejnit" poradi hodnot v targetu a faktorech
data_stacked = pd.concat(objs=[factors_data, target], axis="columns").stack(level=0).swaplevel(i=0, j=1, axis="index").sort_index(axis="index")
data_stacked = data_stacked.dropna(how="all", subset=["Return"], axis="index")
# .. a vzit zpet target
target_stacked = data_stacked.loc[:, "Return"] # "kratky" target 

### nahrazeni NaNu v prediktorech
factors_stacked = factors_stacked.fillna(value=nan_val_replace_predictor, axis="index", method=None) # "dlouhe" prediktory

### vyfiltrovani vysoce korelovanych promennych s nizsi performance - step 5
predictors_perf = fm_perf_stats.loc[factors_selected, "lport_T_pval"]
predictors_filtered, lowcorr_vars, predictors_corr = fs.remove_correlated(factors_stacked, predictors_perf, maxCorr)

if len(predictors_filtered.columns)==1: # pro jistotu: kdyby zbyla jedina feature
    predictors_final = predictors_filtered.to_frame()
else:
    predictors_final = predictors_filtered
target_final = target_stacked

### Extend the predictor set with polynomial terms
# ... to be done maybe later ???? v modelu od M. Ficury nevychazely jako vyznamny
# https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features

### u modelu s binarnim targetem a kategorickyma features je pak jeste dalsi transformace a preselekce 
### STEP 6.C. BINARY-TARGET MODELS (with WoE-ization and binning)
if model_type in models_categoric:
    target_final = (target_stacked>0).astype(int) # binary target
    # numericky target v techto modelech nebude

    # kdyz by nezbyly zadny promenny po nekterem z filtracnich kroku - vypis a ukonceni fce
    predictors_woe_filtered, predictors_cat_filtered = fs.features_binning_filtering(predictors_filtered, target_stacked, target_final, outsample_start, 
                                                        nQ_woe, nQuant, pValSelect, pConf, pKeep, min_pos, min_nonpos, target_class, maxCorr_cat)
    # => woeizovany a kategorizovany features, dal M. Ficura pouziva woeizovany
    if len(predictors_woe_filtered.columns)==1: # pro jistotu: kdyby zbyla jedina feature
        predictors_final = predictors_woe_filtered.to_frame()
    else:
        predictors_final = predictors_woe_filtered
        
# target muze zacinat drive nez ponechane prediktory -> adekvatne oriznout zacatek targetu
all_data = pd.concat(objs=[predictors_final, target_final], axis="columns")
all_data = all_data.dropna(how="all", subset=predictors_final.columns) # oriznuti podle prediktoru
target_final = all_data.loc[:, "Return"] 
#predictors_final = all_data.loc[:, predictors_final.columns] # predictors_final zustavaji stejne

### rozdeleni na in-/out-sample pro trenovani modelu
predictors_in = predictors_final.loc[:, :outsample_start, :]
predictors_out = predictors_final.loc[:, outsample_start:, :] 
target_in = target_final.loc[:, :outsample_start, :]
target_out = target_final.loc[:, outsample_start:, :] # stejne dlouhy jak predictors_out, ale s NaNem na konci

### priprava out-sample pro backtest modelu (in-sample bude stejny): v out-samplu nesmi byt NaNy ani v targetu
data_out_bt = pd.concat(objs=[predictors_out, target_out], axis="columns").dropna(how="all", subset=["Return"], axis="index")
predictors_out_bt = data_out_bt.drop("Return", axis="columns") 
target_out_bt = data_out_bt.loc[:, "Return"] 

#%%
###############################################################################
#
#                           PREDICTIVE MODELLING
#
###############################################################################

#% MODEL 1: LINEAR RIDGE REGRESSION (CONTINUOUS TARGET)
n_cv = 10 # nCVali - number of crossvalidation samples

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, KFold


def train_linreg_ridge(predictors_train: pd.DataFrame, target_train: pd.Series, n_folds: int):
    """
    My_LinReg_Ridge_CV - Matlab
    
    Finds and trains best linear Ridge regression model according to mean squared error

    Parameters
    ----------
    predictors_train : pd.DataFrame
        evolution of one categoric/quantiled feature for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    target_train : pd.Series
        evolution of target for all assets in portfolio in long/stacked form (nDays*nAssets rows x 1 column)
        
    n_folds : int
        number of folds in cross-validation

    Returns
    -------
    best_mdl : object
        instance of fitted Ridge regression model

    """
    
    reg_alphas = 0.5*(1.5**np.arange(1,30+1)) # rVec

    estim = linear_model.Ridge(fit_intercept=True, normalize=False, solver="auto")
    
    tuned_params = [ {"alpha": reg_alphas} ]
    
    crossval = KFold(n_splits=n_folds, shuffle=False) # bez nahodneho michani dat => bez random seed
    
    # scorer bude MSE (resp. neg. MSE): MSEMat(i,1)=sum((Y-YF).^2); ... [im,ii] = max(-MSEMat);
    # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    search = GridSearchCV(
        estimator=estim, param_grid=tuned_params, scoring="neg_mean_squared_error", cv=crossval,
        n_jobs=-1
    )
    
    search_results = search.fit(X=predictors_train, y=target_train)

    # diagnostika trenovani - zatim neresit
    #best_params = search_results.best_params_
    #cv_results = search_results.cv_results_ # vytahnout z toho nektere vysledky (vyvoj alfy a skore) a vyplotit
    # rezidua atd.

    best_mdl = search_results.best_estimator_
    
    return best_mdl

#warnings.filterwarnings("default")
best_mdl = train_linreg_ridge(predictors_in, target_in, n_cv)

##%% FACTOR MODEL APPLICATION
# predikce na full-samplu
target_fcast = best_mdl.predict(X=predictors_final)
target_fcast = pd.DataFrame(data=target_fcast, index=predictors_final.index).unstack(level=0)
target_fcast.columns = target_fcast.columns.droplevel(level=0) # HM_YF
# vytahnout ten vysledek - predikce: posledni hodnoty (M.Ficura to ma ve StockPicking_ComputeForecasts)
target_fcast_final = target_fcast.loc[target_fcast.index[-1], :] # FINAL SCORE - bla5 ve StockPicking_ComputeForecasts.m
target_fcast_final_date = target_fcast_final.name
target_fcast_final.name = "Score"
print("Assets' score report - {}:".format(freq_rebalanc))
print(target_fcast_final.sort_values(ascending=False).to_frame()*100000)
print("Valid from ", target_fcast_final_date + pd.Timedelta(days=1))

# vytahnout "Score Decomposed" (M.Ficura to ma ve StockPicking_ComputeForecasts)
best_mdl_const = best_mdl.intercept_
best_mdl_betas = pd.Series(data=best_mdl.coef_, index=predictors_in.columns)
predictors_last = predictors_final.loc[:, target_fcast.index[-1], :] # bla4 ve StockPicking_ComputeForecasts.m
target_fcast_decomposed = predictors_last.multiply(other=best_mdl_betas, axis="columns")
target_fcast_decomposed["Constant"] = best_mdl_const
# konstantu na zacatek
target_fcast_decomposed = pd.concat(objs=[target_fcast_decomposed["Constant"], target_fcast_decomposed.drop(labels="Constant", axis="columns")],
                                    axis="columns") # DECOMPOSED SCORE - bla6 ve StockPicking_ComputeForecasts.m (+ konstanta)
#target_fcast_decomposed_fin = target_fcast_decomposed.sum(axis="columns") # jen pro kontrolu s target_fcast_final

#return target_fcast_final, target_fcast_decomposed


##% Compute out-sample portfolio performance stats
target_all_out = target_all.loc[outsample_start:, :] # neshiftnuty target
target_fcast_out = target_fcast.loc[outsample_start:target_all_out.index[-1], :] # jen po 1. predikovany pozorovani
# f-m returns pro forecast returnu (pro vsechny assety)
fm_ret, fm_ret_cum, fm_ret_diff, fm_ret_diff_cum, fm_corr, nRand = fs.factor_mimicking_returns(target_fcast_out, target_all_out, nQuant)

# uprava promennych (multiindexy atd.), aby sly pouzit v portfolio_perf_stats
fm_ret_diff_cum = fm_ret_diff_cum.to_frame()
fm_ret_diff_cum.columns = ["Return_fcast"]
fm_corr = fm_corr.to_frame()
fm_corr.columns = ["Return_fcast"]
fm_ret_cum.columns = pd.MultiIndex.from_product(iterables=[ ["Return_fcast"], list(fm_ret_cum.columns) ]) # pridani levelu

fm_perf_stats_fcast = fs.portfolio_perf_stats(fm_ret_diff_cum, fm_ret_cum, fm_corr, kYear) # FM_Stats_YF [1 x 16 statistik]

#%%
###############################################################################
#
#                           BACKTESTING
#
###############################################################################
""" ValueError: Input contains NaN, infinity or a value too large for dtype('float64')."""
stockpicker_pnl, hodl_pnl = backtest_stockpicker(
                model=best_mdl,
                df=df,
                X_train=predictors_in,
                y_train=target_in,
                X_test=predictors_out_bt, 
                y_test=target_out_bt,
                #start_date=outsample_start,
                start_date='2021-01-01',
                freq_rebalanc=freq_rebalanc,
                desired_stocks=tickers,
                leverage=1.3,
                #unwanted_stocks=["AAPL"],
                plot=False,
                number_assets=3
                )
"""
# SOLVED (inconsistency in dataframes where some of then used weekends and other did not) 
# KeyError: Timestamp('2018-03-31 00:00:00')
# problem  v cyklu for idx in timeSerie_test: na radku 211
# stockpickerLong_pnl = stockpickerLong_pnl.append(pd.Series(return_no_scale.loc[idx, long.index].mean())) # idx is a specific day, long.index is for appropriate stocks
"""

#%%
###############################################################################
#
#                           DIAGNOSTICS
#
###############################################################################
"""
print_metrics(model=best_mdl,
              X_test=predictors_out_bt,
              y_test=target_out_bt)

plot_learning_curve(estimator=linear_model.Ridge(alpha=best_mdl.alpha, fit_intercept=True, normalize=False, solver="auto"),
     X_train=predictors_in,
     y_train=target_in)
"""
#%%

end_time = time.time()
print("ELAPSED TIME: ", end_time - start_time) # measuring elapsed time






