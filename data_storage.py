
import data_computer as dc
import feature_selection as fs

import yfinance as yf
import pandas as pd

import re
import os
import time
import requests


"""
# funkce na obhospodarovani ziskani cenovych dat
# .. primarne se hleda ulozeny .parquet soubor na disku
# .. kdyz je problem (neexistuje, nejd eotevrit apod.), stahuje se z YF

# funkce pro obhospodarovani ulozenych faktoru - zakladni "daily" varianta
# .. pocitaji i ukladaji se vzdy jen daily - dle dalsich potreb (frekv. rebalance atd.) se pak prevzorkuje
# .. "zabaluje" (pouziva) funkce pro vypocet faktoru (z data_computer) - zmena v dc muze zpusobit problem zde v ds
# .. kontrolovano pro jednotlive assety (assety vytazeny ze vstupnich OHLCV dat)

# funkce pro obhospodarovani ulozenych faktor-mimicking portfolio returns
# .. pocitaji i ukladaji se vzdy dle zadane granularity a asset univerza
# .. "zabaluje" (pouziva) funkci pro vypocet f-m returns (z feature_selection) - zmena ve fs muze zpusobit problem zde v ds
"""




#%%
##########################
# SAMPLE-DATA
##########################
def get_sp100(base_url='https://app.analyticalplatform.com/api/sp/Ticker?page_size=200&sp100=true'):
    '''
    Return list of S&P100 constituents.

    Parameters
    ----------
    base_url : string, optional
        The default is 'https://app.analyticalplatform.com/api/sp/Ticker?page_size=200&sp100=true'.

    Returns
    -------
    tickers : list
        List of tickers.

    '''
    start_time = time.time()
    r = requests.get(base_url)
    response_list = r.json()
    
    tickers = []
    for i in range(len(response_list['results'])):
        ticker = response_list['results'][i]['symbol']
        tickers.append(ticker)
    
    end_time = time.time()
    print("Elapsed Time for Task 'RECEIVE SP100 TICKERS': ", end_time - start_time)
    return tickers

#sp100 = get_sp100()




def get_ohlcv_data(tickers: list, start_date: str, SAVED_data_dir: str):    
    '''
    Read data from disc. If any problem with reading data from disc, it is downloaded from yahoo finance.
    Output dataset is hiearchical in Ticker-Specific Data Serie order. 

    Parameters
    ----------
    tickers : list
        List of desired tickers for fetching data.
    start_date : string
        Output data is cropped so that history of the whole dataset starts with 'start_date' value.
    SAVED_data_dir : str
        basic directory for storing another groups of data

    Returns
    -------
    data : dataframe
        Output price hiearchical dataset.

    '''
    
    DIR_ohlcv = os.path.join(SAVED_data_dir, "price_data")

    data = pd.DataFrame()
    for ticker in tickers:
        path = os.path.join(DIR_ohlcv, "{}-f1.parquet".format(ticker))
        #path = "../SP-data/price_data/{}-f1.parquet".format(ticker)
        #print(path)
        try:
            data_tmp = pd.read_parquet(path=path)
        except: # pri problemu s nactenim z disku stahnout z YF
            print("Problem reading data from ", path, "... downloading from YF")
            yf_ticker = yf.Ticker(ticker)
            data_tmp = yf_ticker.history(period="1d", start=start_date, auto_adjust=True)
            
        data_tmp.columns = pd.MultiIndex.from_product(iterables=[[ticker], data_tmp.columns])
        data = pd.concat(objs=[data, data_tmp], axis='columns')
        data = data.loc[start_date:, :]
    return data



def get_statistical_moments(df_ohlcv: pd.DataFrame, horizons: list, SAVED_data_dir: str):
    """
    Managing storage, usage and computing of statistical features.

    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        input DataFrame with "raw" prices etc. data
    horizons : list
        lenght of horizons for features calculation
    SAVED_data_dir : str
        basic directory with other sub-directories for storing data of specific groups of features
    
    Returns
    -------
    factors_daily_alltickers : pd.DataFrame
        statistical moments predictors corresponding to the ohlcv data (all tickers, all days)
    """

    tickers = pd.unique( pd.DataFrame(data=list(df_ohlcv.columns))[0] )

    factors_title = "Statistical moments factors"
    DIR_factors = os.path.join(SAVED_data_dir, "stat_moments")
    #path_factors_daily = os.path.join(DIR_factors, "stat_moments.parquet")
    if (not os.path.exists(DIR_factors)):  # create dir if it does not exist
        os.makedirs(DIR_factors)
        print(factors_title, " - directory for using precomputed data created: ", os.path.abspath(DIR_factors))
        
        # spocitat cele (a pak ulozit)
        print(factors_title, " - computing all horizons completely ...")
        factors_daily_alltickers = dc.statistical_moments(df_ohlcv, horizons)  
        
        # ... a ulozit - prepsat aktualnimi kompletnimi daty
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_stat_moments.parquet".format(ticker))
            factors_daily_t = factors_daily_alltickers.loc[:, (ticker, slice(None))]
            factors_daily_t.to_parquet(path_factors_daily) # info o columns se ulozi dobre (bez puvodnich polozek v nejvyssim levelu)
            
    else: # slozka pro faktory existuje
    
        factors_daily_alltickers = pd.DataFrame()
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_stat_moments.parquet".format(ticker))
            
            df_ohlcv_t = df_ohlcv.loc[:, (ticker, slice(None))]
            # problem v compute_return() v statistical_moments(): ValueError: Shape of passed values is (69, 1), indices imply (69, 7)
            # => predelat columns na spravny, zustavaji tam puvodni polozky v nejvyssim levelu (vsechny tickery)
            df_ohlcv_t.columns = pd.MultiIndex.from_product(iterables=[[ticker], df_ohlcv_t.columns.levels[1] ])
                
            # nacist a/nebo dopocitat pro dany ticker
            if os.path.exists(path_factors_daily):
                print(factors_title, " - reading precomputed for {} ...".format(ticker))
                factors_daily_tmp = pd.read_parquet(path_factors_daily) # index zustane datetime
                
                # zjistit, u jakych horizontu pripadne chybi "nejnovejsi" pozorovani    
                factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_tmp.columns))[1] ))
                horiz_tmp = set([int(re.findall("\\d+", s)[-1]) for s in factors_tmp]) # potencialne nekompletni horizonty
                # posledni prvek - kdyby se pridaval nejaky nazev obsahujici cislo
                
                factors_daily_new = pd.DataFrame()
                # pro kazdy potencialne nekompletni horizont resit individualne
                for h in horiz_tmp:
                    df_ohlcv_tmp = df_ohlcv_t.loc[ factors_daily_tmp.index[-h]:] # zacatek zrojovych dat o 1 den drive nez horizont od konce
                    factors_daily_new_h = pd.DataFrame() # inicializace - aby se pripadne neconcatil df z minule iterace
                    if (len(df_ohlcv_tmp)-h)>0: # kdyz neco chybelo
                        print(factors_title, " - computing missing observations for horizon {}...".format(h))
                        factors_daily_new_h = dc.statistical_moments(df_ohlcv_tmp, [h])
                        factors_daily_new_h = factors_daily_new_h.loc[factors_daily_tmp.index[-1]:] # nove zacinaji o 1 driv nez zacatek chybejicich
                        
                    factors_daily_new = pd.concat(objs=[factors_daily_new, factors_daily_new_h], axis="columns")
                
                if len(factors_daily_new)>1: # delka jen 1 = neni nic noveho (zacatek novych o 1 drive nez 1. chybejici)
                    factors_daily_all = pd.concat(objs=[factors_daily_tmp.loc[:, (ticker, factors_tmp)], # jen features ze souboru
                                                     factors_daily_new.iloc[1:] ], axis="index") # od 2. pozorovani        
                else: # kdyz nic nechybelo - zustane to nactene
                    print(factors_title, " - no new observations: using only stored ones ...")
                    factors_daily_all = factors_daily_tmp.loc[:, (slice(None), factors_tmp)]            
                
                # uplne chybejici horizonty/features
                horiz_mis = list(set(horizons) - set(horiz_tmp)) 
                if len(horiz_mis)>0: # kdyz nejake horizonty chybely kompletne
                    print(factors_title, " - computing whole missing horizons {} ...".format(horiz_mis))
                    factors_daily_new = dc.statistical_moments(df_ohlcv_t, horiz_mis)
                else: # kdyz nic nechybelo - zustanou prazdny df
                    print(factors_title, " - no horizons missing completely (all observations) ...")
                    factors_daily_new = pd.DataFrame()
                    
                # spojit dopocitane casti pro dany ticker (ulozit pozdeji)
                factors_daily = pd.concat(objs=[factors_daily_all, factors_daily_new], axis="columns").sort_index(axis="columns")
            
            else: # soubor s daty pro dany ticker neexistuje - spocitat pro nej vsechny horizonty/features
                print(factors_title, " - computing all horizons completely for {} ...".format(ticker))
                factors_daily = dc.statistical_moments(df_ohlcv_t, horizons)
            
            # ulozit hned po spocitani, kdyby to nekde exlo at se neztrati vse
            factors_daily.to_parquet(path_factors_daily)
            
            # spojit dopocitane casti pro vsechny tickery
            factors_daily_alltickers = pd.concat(objs=[factors_daily_alltickers, factors_daily], axis="columns").sort_index(axis="columns")
    
    """
    # TEST - ulozit, aby neco chybelo
    factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_alltickers.columns))[1] ))
    horiz_factors = [s for s in factors_tmp if ("22" in s)]
    
    #factors_daily.iloc[:-3].drop(labels=horiz_factors, axis="columns", level=1).to_parquet(path_factors)
    #factors_daily.iloc[:-3].drop(labels=horiz_factors, axis="columns", level=1).to_csv(path_factors.replace(".parquet", ".csv"))
    
    for ticker in tickers: 
        path_factors_daily = os.path.join(DIR_factors, "{}_stat_moments.parquet".format(ticker))
        factors_daily_tmp = factors_daily_alltickers.iloc[:-3].drop(labels=horiz_factors, axis="columns", level=1).loc[:, (ticker, slice(None))]
        factors_daily_tmp.to_parquet(path_factors_daily)

    """
    return factors_daily_alltickers



def get_price_technical(df_ohlcv: pd.DataFrame, horizons: list, SAVED_data_dir: str):
    """
    Managing storage, usage and computing of technical features based on price data.
    Nearly identical to get_statistical_moments() but needs special treatment for TRANGE (without horizon)

    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        input DataFrame with "raw" prices etc. data
    horizons : list
        lenght of horizons for features calculation
    SAVED_data_dir : str
        basic directory with other sub-directories for storing data of specific groups of features
        
    Returns
    -------
    factors_daily_alltickers : pd.DataFrame
        price technical predictors corresponding to the ohlcv data (all tickers, all days)
    """

    tickers = pd.unique( pd.DataFrame(data=list(df_ohlcv.columns))[0] )

    factors_title = "Price technical factors"
    DIR_factors = os.path.join(SAVED_data_dir, "price_technical")
    #path_factors_daily = os.path.join(DIR_factors, "stat_moments.parquet")
    if (not os.path.exists(DIR_factors)):  # create dir if it does not exist
        os.makedirs(DIR_factors)
        print(factors_title, " - directory for using precomputed data created: ", os.path.abspath(DIR_factors))
        
        # spocitat cele (a pak ulozit)
        print(factors_title, " - computing all horizons completely ...")
        factors_daily_alltickers = dc.technical_factors(df_ohlcv, horizons)  
        
        # ... a ulozit - prepsat aktualnimi kompletnimi daty
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_price_tech.parquet".format(ticker))
            factors_daily_t = factors_daily_alltickers.loc[:, (ticker, slice(None))]
            factors_daily_t.to_parquet(path_factors_daily) # info o columns se ulozi dobre (bez puvodnich polozek v nejvyssim levelu)
        
    else: # slozka pro faktory existuje
    
        factors_daily_alltickers = pd.DataFrame()
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_price_tech.parquet".format(ticker))
            
            df_ohlcv_t = df_ohlcv.loc[:, (ticker, slice(None))]
            # => predelat columns na spravny, zustavaji tam puvodni polozky v nejvyssim levelu (vsechny tickery)
            df_ohlcv_t.columns = pd.MultiIndex.from_product(iterables=[[ticker], df_ohlcv_t.columns.levels[1] ])
                
            # nacist a/nebo dopocitat pro dany ticker
            if os.path.exists(path_factors_daily):
                print(factors_title, " - reading precomputed for {} ...".format(ticker))
                factors_daily_tmp = pd.read_parquet(path_factors_daily) # index zustane datetime
                
                # zjistit, u jakych horizontu pripadne chybi "nejnovejsi" pozorovani    
                factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_tmp.columns))[1] ))
                # TRANGE nema horizont -> problem pri detekci delek horizontu IndexError: list index out of range 
                factors_tmp.remove("TRANGE") # vyhodit ze seznamu (bude se ale objevovat opakovane pro kazdy horizont)
                horiz_tmp = set([int(re.findall("\\d+", s)[-1]) for s in factors_tmp]) # potencialne nekompletni horizonty
                # posledni prvek - nektere nazvy maji v sobe cislo nevazici se na vstupni horizont
                
                factors_daily_new = pd.DataFrame()
                # pro kazdy potencialne nekompletni horizont resit individualne
                for h in horiz_tmp:
                    df_ohlcv_tmp = df_ohlcv_t.loc[ factors_daily_tmp.index[-h]:] # zacatek zrojovych dat o 1 den drive nez horizont od konce
                    factors_daily_new_h = pd.DataFrame() # inicializace - aby se pripadne neconcatil df z minule iterace
                    if (len(df_ohlcv_tmp)-h)>0: # kdyz neco chybelo
                        print(factors_title, " - computing missing observations for horizon {}...".format(h))
                        factors_daily_new_h = dc.technical_factors(df_ohlcv_tmp, [h])
                        factors_daily_new_h = factors_daily_new_h.loc[factors_daily_tmp.index[-1]:] # nove zacinaji o 1 driv nez zacatek chybejicich
                        
                    factors_daily_new = pd.concat(objs=[factors_daily_new, factors_daily_new_h], axis="columns")
                # je tam vicekrat ten TRANGE -> vyhodit - byl by problem s concatenim indexu
                if factors_daily_new.columns.duplicated().sum()>0:
                    #print("Duplicate names in features ", factors_daily_new.columns[factors_daily_new.columns.duplicated()] )
                    factors_daily_new = factors_daily_new.loc[:, ~factors_daily_new.columns.duplicated()] # odstraneni duplikatu
                
                factors_tmp.append("TRANGE") # aby pak nechybel ve vyberu (a po concatu nevznikla prazdna cast sloupce)
                if len(factors_daily_new)>1: # delka jen 1 = neni nic noveho (zacatek novych o 1 drive nez 1. chybejici)
                    factors_daily_all = pd.concat(objs=[factors_daily_tmp.loc[:, (ticker, factors_tmp)], # jen features ze souboru
                                                     factors_daily_new.iloc[1:] ], axis="index") # od 2. pozorovani        
                else: # kdyz nic nechybelo - zustane to nactene
                    print(factors_title, " - no new observations: using only stored ones ...")
                    factors_daily_all = factors_daily_tmp.loc[:, (slice(None), factors_tmp)]            
                
                # uplne chybejici horizonty/features
                horiz_mis = list(set(horizons) - set(horiz_tmp)) 
                if len(horiz_mis)>0: # kdyz nejake horizonty chybely kompletne
                    print(factors_title, " - computing whole missing horizons {} ...".format(horiz_mis))
                    factors_daily_new = dc.technical_factors(df_ohlcv_t, horiz_mis)
                else: # kdyz nic nechybelo - zustanou prazdny df
                    print(factors_title, " - no horizons missing completely (all observations) ...")
                    factors_daily_new = pd.DataFrame()
                    
                # spojit dopocitane casti pro dany ticker (ulozit pozdeji)
                factors_daily = pd.concat(objs=[factors_daily_all, factors_daily_new], axis="columns").sort_index(axis="columns")
                # znova tam je vicekrat ten TRANGE -> vyhodit 
                if factors_daily.columns.duplicated().sum()>0:
                    #print("Duplicate names in features ", factors_daily.columns[factors_daily.columns.duplicated()] )
                    factors_daily = factors_daily.loc[:, ~factors_daily.columns.duplicated()] # odstraneni duplikatu
                    
            else: # soubor s daty pro dany ticker neexistuje - spocitat pro nej vsechny horizonty/features
                print(factors_title, " - computing all horizons completely for {} ...".format(ticker))
                factors_daily = dc.technical_factors(df_ohlcv_t, horizons)
            
            # ulozit hned po spocitani, kdyby to nekde exlo at se neztrati vse
            factors_daily.to_parquet(path_factors_daily)
            
            # spojit dopocitane casti pro vsechny tickery
            factors_daily_alltickers = pd.concat(objs=[factors_daily_alltickers, factors_daily], axis="columns").sort_index(axis="columns")
    
    """
    # TEST - ulozit, aby neco chybelo
    factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_alltickers.columns))[1] ))
    horiz_factors = [s for s in factors_tmp if ("22" in s)]
        
    for ticker in tickers: 
        path_factors_daily = os.path.join(DIR_factors, "{}_price_tech.parquet".format(ticker))
        factors_daily_tmp = factors_daily_alltickers.iloc[:-3].drop(labels=horiz_factors, axis="columns", level=1).loc[:, (ticker, slice(None))]
        factors_daily_tmp.to_parquet(path_factors_daily)

    """
    return factors_daily_alltickers



def get_volume_technical(df_ohlcv: pd.DataFrame, horizons: list, SAVED_data_dir: str):
    """
    Managing storage, usage and computing of technical features based on volume data.
    Identical to get_statistical_moments() - no special treatment for indicators without horizon (as in get_price_technical())
    
    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        input DataFrame with "raw" prices etc. data
    horizons : list
        lenght of horizons for features calculation
    SAVED_data_dir : str
        basic directory with other sub-directories for storing data of specific groups of features

    Returns
    -------
    factors_daily_alltickers : pd.DataFrame
        volume technical predictors corresponding to the ohlcv data (all tickers, all days)
    """

    tickers = pd.unique( pd.DataFrame(data=list(df_ohlcv.columns))[0] )

    factors_title = "Volume technical factors"
    DIR_factors = os.path.join(SAVED_data_dir, "volume_technical")
    #path_factors_daily = os.path.join(DIR_factors, "stat_moments.parquet")
    if (not os.path.exists(DIR_factors)):  # create dir if it does not exist
        os.makedirs(DIR_factors)
        print(factors_title, " - directory for using precomputed data created: ", os.path.abspath(DIR_factors))
        
        # spocitat cele (a pak ulozit)
        print(factors_title, " - computing all horizons completely ...")
        factors_daily_alltickers = dc.volume_technical_factors(df_ohlcv, horizons)  
        
        # ... a ulozit - prepsat aktualnimi kompletnimi daty
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_vol_tech.parquet".format(ticker))
            factors_daily_t = factors_daily_alltickers.loc[:, (ticker, slice(None))]
            factors_daily_t.to_parquet(path_factors_daily) # info o columns se ulozi dobre (bez puvodnich polozek v nejvyssim levelu)
        
    else: # slozka pro faktory existuje
    
        factors_daily_alltickers = pd.DataFrame()
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_vol_tech.parquet".format(ticker))
            
            df_ohlcv_t = df_ohlcv.loc[:, (ticker, slice(None))]
            # problem v compute_return() v statistical_moments(): ValueError: Shape of passed values is (69, 1), indices imply (69, 7)
            # => predelat columns na spravny, zustavaji tam puvodni polozky v nejvyssim levelu (vsechny tickery)
            df_ohlcv_t.columns = pd.MultiIndex.from_product(iterables=[[ticker], df_ohlcv_t.columns.levels[1] ])
                
            # nacist a/nebo dopocitat pro dany ticker
            if os.path.exists(path_factors_daily):
                print(factors_title, " - reading precomputed for {} ...".format(ticker))
                factors_daily_tmp = pd.read_parquet(path_factors_daily) # index zustane datetime
                
                # zjistit, u jakych horizontu pripadne chybi "nejnovejsi" pozorovani    
                factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_tmp.columns))[1] ))
                horiz_tmp = set([int(re.findall("\\d+", s)[-1]) for s in factors_tmp]) # potencialne nekompletni horizonty
                # posledni prvek - kdyby se pridaval nejaky nazev obsahujici cislo
                
                factors_daily_new = pd.DataFrame()
                # pro kazdy potencialne nekompletni horizont resit individualne
                for h in horiz_tmp:
                    df_ohlcv_tmp = df_ohlcv_t.loc[ factors_daily_tmp.index[-h]:] # zacatek zrojovych dat o 1 den drive nez horizont od konce
                    factors_daily_new_h = pd.DataFrame() # inicializace - aby se pripadne neconcatil df z minule iterace
                    if (len(df_ohlcv_tmp)-h)>0: # kdyz neco chybelo
                        print(factors_title, " - computing missing observations for horizon {}...".format(h))
                        factors_daily_new_h = dc.volume_technical_factors(df_ohlcv_tmp, [h])
                        factors_daily_new_h = factors_daily_new_h.loc[factors_daily_tmp.index[-1]:] # nove zacinaji o 1 driv nez zacatek chybejicich
                        
                    factors_daily_new = pd.concat(objs=[factors_daily_new, factors_daily_new_h], axis="columns")
                
                if len(factors_daily_new)>1: # delka jen 1 = neni nic noveho (zacatek novych o 1 drive nez 1. chybejici)
                    factors_daily_all = pd.concat(objs=[factors_daily_tmp.loc[:, (ticker, factors_tmp)], # jen features ze souboru
                                                     factors_daily_new.iloc[1:] ], axis="index") # od 2. pozorovani        
                else: # kdyz nic nechybelo - zustane to nactene
                    print(factors_title, " - no new observations: using only stored ones ...")
                    factors_daily_all = factors_daily_tmp.loc[:, (slice(None), factors_tmp)]            
                
                # uplne chybejici horizonty/features
                horiz_mis = list(set(horizons) - set(horiz_tmp)) 
                if len(horiz_mis)>0: # kdyz nejake horizonty chybely kompletne
                    print(factors_title, " - computing whole missing horizons {} ...".format(horiz_mis))
                    factors_daily_new = dc.volume_technical_factors(df_ohlcv_t, horiz_mis)
                else: # kdyz nic nechybelo - zustanou prazdny df
                    print(factors_title, " - no horizons missing completely (all observations) ...")
                    factors_daily_new = pd.DataFrame()
                    
                # spojit dopocitane casti pro dany ticker (ulozit pozdeji)
                factors_daily = pd.concat(objs=[factors_daily_all, factors_daily_new], axis="columns").sort_index(axis="columns")
            
            else: # soubor s daty pro dany ticker neexistuje - spocitat pro nej vsechny horizonty/features
                print(factors_title, " - computing all horizons completely for {} ...".format(ticker))
                factors_daily = dc.volume_technical_factors(df_ohlcv_t, horizons)
            
            # ulozit hned po spocitani, kdyby to nekde exlo at se neztrati vse
            factors_daily.to_parquet(path_factors_daily)
            
            # spojit dopocitane casti pro vsechny tickery
            factors_daily_alltickers = pd.concat(objs=[factors_daily_alltickers, factors_daily], axis="columns").sort_index(axis="columns")
        
    """
    # TEST - ulozit, aby neco chybelo
    factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_alltickers.columns))[1] ))
    horiz_factors = [s for s in factors_tmp if ("22" in s)]
       
    for ticker in tickers: 
        path_factors_daily = os.path.join(DIR_factors, "{}_vol_tech.parquet".format(ticker))
        factors_daily_tmp = factors_daily_alltickers.iloc[:-3].drop(labels=horiz_factors, axis="columns", level=1).loc[:, (ticker, slice(None))]
        factors_daily_tmp.to_parquet(path_factors_daily)

    """
    return factors_daily_alltickers



def get_ar1_logprice(df_ohlcv: pd.DataFrame, horizons: list, SAVED_data_dir: str):
    """
    Managing storage, usage and computing of ar1 features based on logarithmic price data.
    Nearly identical to get_statistical_moments()
    (but because of very demanding calculation, the first/initial computation is performed ticker-by-ticker)

    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        input DataFrame with "raw" prices etc. data
    horizons : list
        lenght of horizons for features calculation
    SAVED_data_dir : str
        basic directory with other sub-directories for storing data of specific groups of features

    Returns
    -------
    factors_daily_alltickers : pd.DataFrame
        ar1 logarithmic price predictors corresponding to the ohlcv data (all tickers, all days)
    """

    tickers = pd.unique( pd.DataFrame(data=list(df_ohlcv.columns))[0] )

    factors_title = "AR1 factors"
    DIR_factors = os.path.join(SAVED_data_dir, "ar1_logprice")
    #path_factors_daily = os.path.join(DIR_factors, "stat_moments.parquet")
    if (not os.path.exists(DIR_factors)):  # create dir if it does not exist
        os.makedirs(DIR_factors)
        print(factors_title, " - directory for using precomputed data created: ", os.path.abspath(DIR_factors))
        
        #print(factors_title, " - computing all horizons completely ...")
        #factors_daily_alltickers = dc.ar1_factors_weeks(df_ohlcv, horizons)  
        # spocitat cele - radeji po tickerech (velmi pomaly vypocet, kdyby to exlo, at se neztrati vse)
        factors_daily_alltickers = pd.DataFrame()
        for ticker in tickers: 
            print(factors_title, " - computing all horizons completely for {} ...".format(ticker))
            df_ohlcv_t = df_ohlcv.loc[:, (ticker, slice(None))]
            path_factors_daily = os.path.join(DIR_factors, "{}_ar1_logprice.parquet".format(ticker))
            factors_daily_t = dc.ar1_factors(df_ohlcv_t, horizons)              
            factors_daily_t.to_parquet(path_factors_daily) # info o columns se ulozi dobre (bez puvodnich polozek v nejvyssim levelu)
            
            # spojit dopocitane casti pro vsechny tickery
            factors_daily_alltickers = pd.concat(objs=[factors_daily_alltickers, factors_daily_t], axis="columns").sort_index(axis="columns")
        
    else: # slozka pro faktory existuje
    
        factors_daily_alltickers = pd.DataFrame()
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_ar1_logprice.parquet".format(ticker))
            
            df_ohlcv_t = df_ohlcv.loc[:, (ticker, slice(None))]
            # problem v compute_return() v statistical_moments(): ValueError: Shape of passed values is (69, 1), indices imply (69, 7)
            # => predelat columns na spravny, zustavaji tam puvodni polozky v nejvyssim levelu (vsechny tickery)
            df_ohlcv_t.columns = pd.MultiIndex.from_product(iterables=[[ticker], df_ohlcv_t.columns.levels[1] ])
                
            # nacist a/nebo dopocitat pro dany ticker
            if os.path.exists(path_factors_daily):
                print(factors_title, " - reading precomputed for {} ...".format(ticker))
                factors_daily_tmp = pd.read_parquet(path_factors_daily) # index zustane datetime
                
                # zjistit, u jakych horizontu pripadne chybi "nejnovejsi" pozorovani    
                factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_tmp.columns))[1] ))
                horiz_tmp = set([int(re.findall("\\d+", s)[-1]) for s in factors_tmp]) # potencialne nekompletni horizonty
                # posledni prvek - kdyby se pridaval nejaky nazev obsahujici cislo
                #horiz_tmp = [int(s.replace("AR1_LogPrice_" , "")) for s in factors_tmp] # potencialne nekompletni horizonty

                factors_daily_new = pd.DataFrame()
                # pro kazdy potencialne nekompletni horizont resit individualne
                for h in horiz_tmp:
                    df_ohlcv_tmp = df_ohlcv_t.loc[ factors_daily_tmp.index[-h]:] # zacatek zrojovych dat o 1 den drive nez horizont od konce
                    factors_daily_new_h = pd.DataFrame() # inicializace - aby se pripadne neconcatil df z minule iterace
                    if (len(df_ohlcv_tmp)-h)>0: # kdyz neco chybelo
                        print(factors_title, " - computing missing observations for horizon {}...".format(h))
                        factors_daily_new_h = dc.ar1_factors(df_ohlcv_tmp, [h])
                        factors_daily_new_h = factors_daily_new_h.loc[factors_daily_tmp.index[-1]:] # nove zacinaji o 1 driv nez zacatek chybejicich
                        
                    factors_daily_new = pd.concat(objs=[factors_daily_new, factors_daily_new_h], axis="columns")
                
                if len(factors_daily_new)>1: # delka jen 1 = neni nic noveho (zacatek novych o 1 drive nez 1. chybejici)
                    factors_daily_all = pd.concat(objs=[factors_daily_tmp.loc[:, (ticker, factors_tmp)], # jen features ze souboru
                                                     factors_daily_new.iloc[1:] ], axis="index") # od 2. pozorovani        
                else: # kdyz nic nechybelo - zustane to nactene
                    print(factors_title, " - no new observations: using only stored ones ...")
                    factors_daily_all = factors_daily_tmp.loc[:, (slice(None), factors_tmp)]            
                
                # uplne chybejici horizonty/features
                horiz_mis = list(set(horizons) - set(horiz_tmp)) 
                if len(horiz_mis)>0: # kdyz nejake horizonty chybely kompletne
                    print(factors_title, " - computing whole missing horizons {} ...".format(horiz_mis))
                    factors_daily_new = dc.ar1_factors(df_ohlcv_t, horiz_mis)
                else: # kdyz nic nechybelo - zustanou prazdny df
                    print(factors_title, " - no horizons missing completely (all observations) ...")
                    factors_daily_new = pd.DataFrame()
                    
                # spojit dopocitane casti pro dany ticker (ulozit pozdeji)
                factors_daily = pd.concat(objs=[factors_daily_all, factors_daily_new], axis="columns").sort_index(axis="columns")
            
            else: # soubor s daty pro dany ticker neexistuje - spocitat pro nej vsechny horizonty/features
                print(factors_title, " - computing all horizons completely for {} ...".format(ticker))
                factors_daily = dc.ar1_factors(df_ohlcv_t, horizons)
            
            # ulozit hned po spocitani, kdyby to nekde exlo at se neztrati vse
            factors_daily.to_parquet(path_factors_daily)

            # spojit dopocitane casti pro vsechny tickery
            factors_daily_alltickers = pd.concat(objs=[factors_daily_alltickers, factors_daily], axis="columns").sort_index(axis="columns")
        
    """
    # TEST - ulozit, aby neco chybelo
    factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_alltickers.columns))[1] ))
    horiz_factors = [s for s in factors_tmp if ("22" in s)]
      
    for ticker in tickers: 
        path_factors_daily = os.path.join(DIR_factors, "{}_ar1_logprice.parquet".format(ticker))
        factors_daily_tmp = factors_daily_alltickers.iloc[:-3].drop(labels=horiz_factors, axis="columns", level=1).loc[:, (ticker, slice(None))]
        factors_daily_tmp.to_parquet(path_factors_daily)

    """
    return factors_daily_alltickers



def get_ar1_logprice_weeks(df_ohlcv: pd.DataFrame, horizons: list, SAVED_data_dir: str):
    """
    Managing storage, usage and computing of ar1 weeks features based on logarithmic price data.
    Nearly identical to get_statistical_moments() but needs special treatment for lookback window (horizon in weeks)
    (and because of very demanding calculation, the first/initial computation is performed ticker-by-ticker)
    
    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        input DataFrame with "raw" prices etc. data
    horizons : list
        lenght of horizons for features calculation
    SAVED_data_dir : str
        basic directory with other sub-directories for storing data of specific groups of features

    Returns
    -------
    factors_daily_alltickers : pd.DataFrame
        ar1 logarithmic price predictors weeks corresponding to the ohlcv data (all tickers, all days)
    """

    tickers = pd.unique( pd.DataFrame(data=list(df_ohlcv.columns))[0] )

    factors_title = "AR1 factors weeks"
    DIR_factors = os.path.join(SAVED_data_dir, "ar1_logprice_weeks")
    if (not os.path.exists(DIR_factors)):  # create dir if it does not exist
        os.makedirs(DIR_factors)
        print(factors_title, " - directory for using precomputed data created: ", os.path.abspath(DIR_factors))
        
        # spocitat cele - radeji po tickerech (velmi pomaly vypocet, kdyby to exlo, at se neztrati vse)
        factors_daily_alltickers = pd.DataFrame()
        for ticker in tickers: 
            print(factors_title, " - computing all horizons completely for {} ...".format(ticker))
            df_ohlcv_t = df_ohlcv.loc[:, (ticker, slice(None))]
            path_factors_daily = os.path.join(DIR_factors, "{}_ar1_logprice_w.parquet".format(ticker))
            factors_daily_t = dc.ar1_factors_weeks(df_ohlcv_t, horizons)              
            factors_daily_t.to_parquet(path_factors_daily) # info o columns se ulozi dobre (bez puvodnich polozek v nejvyssim levelu)
            
            # spojit dopocitane casti pro vsechny tickery
            factors_daily_alltickers = pd.concat(objs=[factors_daily_alltickers, factors_daily_t], axis="columns").sort_index(axis="columns")
            
    else: # slozka pro faktory existuje
    
        factors_daily_alltickers = pd.DataFrame()
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_ar1_logprice_w.parquet".format(ticker))
            
            df_ohlcv_t = df_ohlcv.loc[:, (ticker, slice(None))]
            # problem v compute_return() v statistical_moments(): ValueError: Shape of passed values is (69, 1), indices imply (69, 7)
            # => predelat columns na spravny, zustavaji tam puvodni polozky v nejvyssim levelu (vsechny tickery)
            df_ohlcv_t.columns = pd.MultiIndex.from_product(iterables=[[ticker], df_ohlcv_t.columns.levels[1] ])
                
            # nacist a/nebo dopocitat pro dany ticker
            if os.path.exists(path_factors_daily):
                print(factors_title, " - reading precomputed for {} ...".format(ticker))
                factors_daily_tmp = pd.read_parquet(path_factors_daily) # index zustane datetime
                
                # zjistit, u jakych horizontu pripadne chybi "nejnovejsi" pozorovani    
                factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_tmp.columns))[1] ))
                horiz_tmp = set([int(re.findall("\\d+", s)[-1]) for s in factors_tmp]) # potencialne nekompletni horizonty
                # posledni prvek - kdyby se pridaval nejaky nazev obsahujici cislo
                #horiz_tmp = [int(s.replace("AR1_LogPrice_", "").replace("_weeks", "")) for s in factors_tmp] # potencialne nekompletni horizonty

                factors_daily_new = pd.DataFrame()
                # pro kazdy potencialne nekompletni horizont resit individualne
                for h in horiz_tmp:
                    df_ohlcv_tmp = df_ohlcv_t.loc[ factors_daily_tmp.index[-h*5]:] # zacatek zrojovych dat o 1 tyden drive nez horizont od konce
                    # kdyby se obchodovalo i o vikendech tak misto *5 bude *7
                    # .. ale nikdy to nemusi byt presne (pocet obchodnich dni v kazdem tydnu) - nekdy bude zdroj dat delsi (ale nikdy kratsi)
                    
                    factors_daily_new_h = pd.DataFrame() # inicializace - aby se pripadne neconcatil df z minule iterace
                    if len(df_ohlcv_tmp.loc[factors_daily_tmp.index[-1]:])>1: # pocet novych dni (1 = posledni den z faktoru je navic/prekryv)
                    #if (len(df_ohlcv_tmp)-h*5)>0: # kdyz neco chybelo???????????? je spravne h*5 ??????
                        print(factors_title, " - computing missing observations for horizon {}...".format(h))
                        factors_daily_new_h = dc.ar1_factors_weeks(df_ohlcv_tmp, [h])
                        factors_daily_new_h = factors_daily_new_h.loc[factors_daily_tmp.index[-1]:] # nove zacinaji o 1 driv nez zacatek chybejicich
                        
                    factors_daily_new = pd.concat(objs=[factors_daily_new, factors_daily_new_h], axis="columns")
                
                if len(factors_daily_new)>1: # delka jen 1 = neni nic noveho (zacatek novych o 1 drive nez 1. chybejici)
                    factors_daily_all = pd.concat(objs=[factors_daily_tmp.loc[:, (ticker, factors_tmp)], # jen features ze souboru
                                                     factors_daily_new.iloc[1:] ], axis="index") # od 2. pozorovani        
                else: # kdyz nic nechybelo - zustane to nactene
                    print(factors_title, " - no new observations: using only stored ones ...")
                    factors_daily_all = factors_daily_tmp.loc[:, (slice(None), factors_tmp)]            
                
                # uplne chybejici horizonty/features
                horiz_mis = list(set(horizons) - set(horiz_tmp)) 
                if len(horiz_mis)>0: # kdyz nejake horizonty chybely kompletne
                    print(factors_title, " - computing whole missing horizons {} ...".format(horiz_mis))
                    factors_daily_new = dc.ar1_factors_weeks(df_ohlcv_t, horiz_mis)
                else: # kdyz nic nechybelo - zustanou prazdny df
                    print(factors_title, " - no horizons missing completely (all observations) ...")
                    factors_daily_new = pd.DataFrame()
                    
                # spojit dopocitane casti pro dany ticker (ulozit pozdeji)
                factors_daily = pd.concat(objs=[factors_daily_all, factors_daily_new], axis="columns").sort_index(axis="columns")
            
            else: # soubor s daty pro dany ticker neexistuje - spocitat pro nej vsechny horizonty/features
                print(factors_title, " - computing all horizons completely for {} ...".format(ticker))
                factors_daily = dc.ar1_factors_weeks(df_ohlcv_t, horizons)
            
            # ulozit hned po spocitani, kdyby to nekde exlo at se neztrati vse
            factors_daily.to_parquet(path_factors_daily)

            # spojit dopocitane casti pro vsechny tickery
            factors_daily_alltickers = pd.concat(objs=[factors_daily_alltickers, factors_daily], axis="columns").sort_index(axis="columns")
        
    """
    # TEST - ulozit, aby neco chybelo
    factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_alltickers.columns))[1] ))
    horiz_factors = [s for s in factors_tmp if ("22" in s)]
      
    for ticker in tickers: 
        path_factors_daily = os.path.join(DIR_factors, "{}_ar1_logprice_w.parquet".format(ticker))
        factors_daily_tmp = factors_daily_alltickers.iloc[:-3].drop(labels=horiz_factors, axis="columns", level=1).loc[:, (ticker, slice(None))]
        factors_daily_tmp.to_parquet(path_factors_daily)

    """
    return factors_daily_alltickers



def get_standardized_unexplained_volume(df_ohlcv: pd.DataFrame, horizons: list, SAVED_data_dir: str):
    """
    Managing storage, usage and computing of standardized unexplained volume features.
    Nearly identical to get_statistical_moments() but needs special treatment for horizons detection (1st horiz. in title is the right one)
    and lookback window (max horizon for suv resid. aggregation hard-written in dc.standardized_unexplained_volume)
    
    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        input DataFrame with "raw" prices etc. data
    horizons : list
        lenght of horizons for features calculation
    SAVED_data_dir : str
        basic directory with other sub-directories for storing data of specific groups of features
    
    Returns
    -------
    factors_daily_alltickers : pd.DataFrame
        SUV predictors corresponding to the ohlcv data (all tickers, all days)
    """

    tickers = pd.unique( pd.DataFrame(data=list(df_ohlcv.columns))[0] )

    factors_title = "SUV factors"
    DIR_factors = os.path.join(SAVED_data_dir, "suv")
    #path_factors_daily = os.path.join(DIR_factors, "stat_moments.parquet")
    if (not os.path.exists(DIR_factors)):  # create dir if it does not exist
        os.makedirs(DIR_factors)
        print(factors_title, " - directory for using precomputed data created: ", os.path.abspath(DIR_factors))
        
        # spocitat cele (a pak ulozit)
        print(factors_title, " - computing all horizons completely ...")
        factors_daily_alltickers = dc.standardized_unexplained_volume(df_ohlcv, horizons)  
        
        # ... a ulozit - prepsat aktualnimi kompletnimi daty
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_suv.parquet".format(ticker))
            factors_daily_t = factors_daily_alltickers.loc[:, (ticker, slice(None))]
            factors_daily_t.to_parquet(path_factors_daily) # info o columns se ulozi dobre (bez puvodnich polozek v nejvyssim levelu)
            
    else: # slozka pro faktory existuje
    
        factors_daily_alltickers = pd.DataFrame()
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_suv.parquet".format(ticker))
            
            df_ohlcv_t = df_ohlcv.loc[:, (ticker, slice(None))]
            # problem v compute_return() v statistical_moments(): ValueError: Shape of passed values is (69, 1), indices imply (69, 7)
            # => predelat columns na spravny, zustavaji tam puvodni polozky v nejvyssim levelu (vsechny tickery)
            df_ohlcv_t.columns = pd.MultiIndex.from_product(iterables=[[ticker], df_ohlcv_t.columns.levels[1] ])
                
            # nacist a/nebo dopocitat pro dany ticker
            if os.path.exists(path_factors_daily):
                print(factors_title, " - reading precomputed for {} ...".format(ticker))
                factors_daily_tmp = pd.read_parquet(path_factors_daily) # index zustane datetime
                
                # zjistit, u jakych horizontu pripadne chybi "nejnovejsi" pozorovani    
                factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_tmp.columns))[1] ))
                horiz_tmp = set([int(re.findall("\\d+", s)[0]) for s in factors_tmp]) # potencialne nekompletni horizonty
                # prvni prvek - dalsi cisla v nazvu jsou agregacni horizonty "natvrdo" (v suffixu)
                
                factors_daily_new = pd.DataFrame()
                # pro kazdy potencialne nekompletni horizont resit individualne
                for h in horiz_tmp:
                    df_ohlcv_tmp = df_ohlcv_t.loc[ factors_daily_tmp.index[(-h-(22*3))]:] # zacatek zrojovych dat o 1 tyden drive nez potrebny horizont od konce
                    # 22*3 - nejdelsi horizont pro agregovane suv resid. ("natvrdo" v dc.standardized_unexplained_volume()) - jinak je malo dat => chybejici sloupce
                    
                    factors_daily_new_h = pd.DataFrame() # inicializace - aby se pripadne neconcatil df z minule iterace
                    if (len(df_ohlcv_tmp)-h)>0: # kdyz neco chybelo
                        print(factors_title, " - computing missing observations for horizon {}...".format(h))
                        factors_daily_new_h = dc.standardized_unexplained_volume(df_ohlcv_tmp, [h])
                        factors_daily_new_h = factors_daily_new_h.loc[factors_daily_tmp.index[-1]:] # nove zacinaji o 1 driv nez zacatek chybejicich
                        
                    factors_daily_new = pd.concat(objs=[factors_daily_new, factors_daily_new_h], axis="columns")
                
                if len(factors_daily_new)>1: # delka jen 1 = neni nic noveho (zacatek novych o 1 drive nez 1. chybejici)
                    factors_daily_all = pd.concat(objs=[factors_daily_tmp.loc[:, (ticker, factors_tmp)], # jen features ze souboru
                                                     factors_daily_new.iloc[1:] ], axis="index") # od 2. pozorovani        
                else: # kdyz nic nechybelo - zustane to nactene
                    print(factors_title, " - no new observations: using only stored ones ...")
                    factors_daily_all = factors_daily_tmp.loc[:, (slice(None), factors_tmp)]            
                
                # uplne chybejici horizonty/features
                horiz_mis = list(set(horizons) - set(horiz_tmp)) 
                if len(horiz_mis)>0: # kdyz nejake horizonty chybely kompletne
                    print(factors_title, " - computing whole missing horizons {} ...".format(horiz_mis))
                    factors_daily_new = dc.standardized_unexplained_volume(df_ohlcv_t, horiz_mis)
                else: # kdyz nic nechybelo - zustanou prazdny df
                    print(factors_title, " - no horizons missing completely (all observations) ...")
                    factors_daily_new = pd.DataFrame()
                    
                # spojit dopocitane casti pro dany ticker (ulozit pozdeji)
                factors_daily = pd.concat(objs=[factors_daily_all, factors_daily_new], axis="columns").sort_index(axis="columns")
            
            else: # soubor s daty pro dany ticker neexistuje - spocitat pro nej vsechny horizonty/features
                print(factors_title, " - computing all horizons completely for {} ...".format(ticker))
                factors_daily = dc.standardized_unexplained_volume(df_ohlcv_t, horizons)
            
            # ulozit hned po spocitani, kdyby to nekde exlo at se neztrati vse
            factors_daily.to_parquet(path_factors_daily)
            
            # spojit dopocitane casti pro vsechny tickery
            factors_daily_alltickers = pd.concat(objs=[factors_daily_alltickers, factors_daily], axis="columns").sort_index(axis="columns")
    
    """
    # TEST - ulozit, aby neco chybelo
    factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_alltickers.columns))[1] ))
    horiz_factors = [s for s in factors_tmp if ("22" in (re.findall("\\d+", s)[0]) )] # jen v 1. vyskytu
        
    for ticker in tickers: 
        path_factors_daily = os.path.join(DIR_factors, "{}_suv.parquet".format(ticker))
        factors_daily_tmp = factors_daily_alltickers.iloc[:-3].drop(labels=horiz_factors, axis="columns", level=1).loc[:, (ticker, slice(None))]
        factors_daily_tmp.to_parquet(path_factors_daily)

    """
    return factors_daily_alltickers


#SAVED_data_dir = "../SP-data/factors_daily" # zakl slozka pro ukladani a cteni dat prediktoru 
def get_standardized_unexplained_volume_weeks(df_ohlcv: pd.DataFrame, horizons: list, SAVED_data_dir: str):
    """
    Managing storage, usage and computing of standardized unexplained volume weeks features.    
    Nearly identical to get_statistical_moments(), but needs special treatment for horizons detection (1st horiz. in title is the right one) 
        and lookback window (horizon in weeks, max horizon for suv resid. aggregation hard-written in dc.standardized_unexplained_volume) 
        as in get_ar1_logprice_weeks()

    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        input DataFrame with "raw" prices etc. data
    horizons : list
        lenght of horizons for features calculation
    SAVED_data_dir : str
        basic directory with other sub-directories for storing data of specific groups of features
    
    Returns
    -------
    factors_daily_alltickers : pd.DataFrame
        SUV predictors weeks corresponding to the ohlcv data (all tickers, all days)
    """

    tickers = pd.unique( pd.DataFrame(data=list(df_ohlcv.columns))[0] )

    factors_title = "SUV factors weeks"
    DIR_factors = os.path.join(SAVED_data_dir, "suv_weeks")
    #path_factors_daily = os.path.join(DIR_factors, "stat_moments.parquet")
    if (not os.path.exists(DIR_factors)):  # create dir if it does not exist
        os.makedirs(DIR_factors)
        print(factors_title, " - directory for using precomputed data created: ", os.path.abspath(DIR_factors))
        
        # spocitat cele (a pak ulozit)
        print(factors_title, " - computing all horizons completely ...")
        factors_daily_alltickers = dc.standardized_unexplained_volume_weeks(df_ohlcv, horizons)  
        
        # ... a ulozit - prepsat aktualnimi kompletnimi daty
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_suv_w.parquet".format(ticker))
            factors_daily_t = factors_daily_alltickers.loc[:, (ticker, slice(None))]
            factors_daily_t.to_parquet(path_factors_daily) # info o columns se ulozi dobre (bez puvodnich polozek v nejvyssim levelu)
            
    else: # slozka pro faktory existuje
    
        factors_daily_alltickers = pd.DataFrame()
        for ticker in tickers: 
            path_factors_daily = os.path.join(DIR_factors, "{}_suv_w.parquet".format(ticker))
            
            df_ohlcv_t = df_ohlcv.loc[:, (ticker, slice(None))]
            # problem v compute_return() v statistical_moments(): ValueError: Shape of passed values is (69, 1), indices imply (69, 7)
            # => predelat columns na spravny, zustavaji tam puvodni polozky v nejvyssim levelu (vsechny tickery)
            df_ohlcv_t.columns = pd.MultiIndex.from_product(iterables=[[ticker], df_ohlcv_t.columns.levels[1] ])
                
            # nacist a/nebo dopocitat pro dany ticker
            if os.path.exists(path_factors_daily):
                print(factors_title, " - reading precomputed for {} ...".format(ticker))
                factors_daily_tmp = pd.read_parquet(path_factors_daily) # index zustane datetime
                
                # zjistit, u jakych horizontu pripadne chybi "nejnovejsi" pozorovani    
                factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_tmp.columns))[1] ))
                horiz_tmp = set([int(re.findall("\\d+", s)[0]) for s in factors_tmp]) # potencialne nekompletni horizonty
                # prvni prvek - dalsi cisla v nazvu jsou agregacni horizonty "natvrdo" (v suffixu)
                
                factors_daily_new = pd.DataFrame()
                # pro kazdy potencialne nekompletni horizont resit individualne
                for h in horiz_tmp:
                    df_ohlcv_tmp = df_ohlcv_t.loc[ factors_daily_tmp.index[(-h-(22*3))*5]:] # zacatek zrojovych dat o 1 tyden drive nez potrebny horizont od konce
                    # 22*3 - nejdelsi horizont pro agregovane suv resid. ("natvrdo" v dc.standardized_unexplained_volume()) - jinak je malo dat => chybejici sloupce
                    # kdyby se obchodovalo i o vikendech tak misto *5 bude *7
                    # .. ale nikdy to nemusi byt presne (pocet obchodnich dni v kazdem tydnu) - nekdy bude zdroj dat delsi (ale nikdy kratsi)
                    
                    factors_daily_new_h = pd.DataFrame() # inicializace - aby se pripadne neconcatil df z minule iterace
                    if len(df_ohlcv_tmp.loc[factors_daily_tmp.index[-1]:])>1: # pocet novych dni (1 = posledni den z faktoru je navic/prekryv)
                    #if (len(df_ohlcv_tmp)-h*5)>0: # kdyz neco chybelo???????????? je spravne h*5 ??????
                        print(factors_title, " - computing missing observations for horizon {}...".format(h))
                        factors_daily_new_h = dc.standardized_unexplained_volume_weeks(df_ohlcv_tmp, [h])
                        factors_daily_new_h = factors_daily_new_h.loc[factors_daily_tmp.index[-1]:] # nove zacinaji o 1 driv nez zacatek chybejicich
                        
                    factors_daily_new = pd.concat(objs=[factors_daily_new, factors_daily_new_h], axis="columns")
                
                if len(factors_daily_new)>1: # delka jen 1 = neni nic noveho (zacatek novych o 1 drive nez 1. chybejici)
                    factors_daily_all = pd.concat(objs=[factors_daily_tmp.loc[:, (ticker, factors_tmp)], # jen features ze souboru
                                                     factors_daily_new.iloc[1:] ], axis="index") # od 2. pozorovani
                                        
                else: # kdyz nic nechybelo - zustane to nactene
                    print(factors_title, " - no new observations: using only stored ones ...")
                    factors_daily_all = factors_daily_tmp.loc[:, (slice(None), factors_tmp)]            
                
                # uplne chybejici horizonty/features
                horiz_mis = list(set(horizons) - set(horiz_tmp)) 
                if len(horiz_mis)>0: # kdyz nejake horizonty chybely kompletne
                    print(factors_title, " - computing whole missing horizons {} ...".format(horiz_mis))
                    factors_daily_new = dc.standardized_unexplained_volume_weeks(df_ohlcv_t, horiz_mis)
                else: # kdyz nic nechybelo - zustanou prazdny df
                    print(factors_title, " - no horizons missing completely (all observations) ...")
                    factors_daily_new = pd.DataFrame()
                    
                # spojit dopocitane casti pro dany ticker (ulozit pozdeji)
                factors_daily = pd.concat(objs=[factors_daily_all, factors_daily_new], axis="columns").sort_index(axis="columns")
            
            else: # soubor s daty pro dany ticker neexistuje - spocitat pro nej vsechny horizonty/features
                print(factors_title, " - computing all horizons completely for {} ...".format(ticker))
                factors_daily = dc.standardized_unexplained_volume_weeks(df_ohlcv_t, horizons)
            
            # ulozit hned po spocitani, kdyby to nekde exlo at se neztrati vse
            factors_daily.to_parquet(path_factors_daily)
            
            # spojit dopocitane casti pro vsechny tickery
            factors_daily_alltickers = pd.concat(objs=[factors_daily_alltickers, factors_daily], axis="columns").sort_index(axis="columns")
    
    """
    # TEST - ulozit, aby neco chybelo
    factors_tmp = list(pd.unique( pd.DataFrame(data=list(factors_daily_alltickers.columns))[1] ))
    horiz_factors = [s for s in factors_tmp if ("22" in (re.findall("\\d+", s)[0]) )] # jen v 1. vyskytu
        
    for ticker in tickers: 
        path_factors_daily = os.path.join(DIR_factors, "{}_suv_w.parquet".format(ticker))
        factors_daily_tmp = factors_daily_alltickers.iloc[:-3].drop(labels=horiz_factors, axis="columns", level=1).loc[:, (ticker, slice(None))]
        factors_daily_tmp.to_parquet(path_factors_daily)

    """
    return factors_daily_alltickers


"""
pozn.: 
    - odvozene fundamental predictors - ulozit zvlast, bez puvodnich dat
    - puvodni fundamentaly - nechat ulozene na puvodnim miste a pouzivat "samostatne"
    
    - OK stat. moments, vol. tech., ar1 - jen horizonty, stejny princip detekce chybejicich horizontu/features
    - NERESENO momentum, currency volume ma horiz. pairs => jiny zpusob detekce chybejicich features nebo spis neresit (dost rychly vypocet)
    - NERESENO cum. volume nema zadny horizonty -  mega rychly vypocet, nema smysl ukladat 
    - OK price tech. - temer stejny princip jako stat. moments - zvlast reseny nehorizontove ukazatele
    - OK ar1-weeks - podobny princip jako u ar1 (stat, vol-tech) az na delku okna divajiciho se zpet
    - NERESENO smoothed_volume_technical_factors ma 2 druhy horizontu => jina detekce chybejicich features (dost rychly vypocet)
    - OK suv a suv-weeks ma vstupni horiz + defaultni horiz pro agregaci => opet jina detekce chybejicich (u -weeks opet jina delka okna dat)
"""




def get_factor_mimicking_returns(features_in: pd.DataFrame, target_in: pd.DataFrame, nQuant: int, SAVED_data_dir: str):
    """
    Wrapper for all_factors_mimicking_returns() - managing storage, usage and computing of possibly missing factor-mimicking portfolio returns. 
    
    Parameters
    ----------
    features_in : pd.DataFrame
        all features for all assets in portfolio (nAssets x nFeatures columns)
    target_in : pd.DataFrame
        returns for all assets in portfolio (nAssets columns) - must be one-level columns' names!!!
    nQuant : int
        number of quantiles the predictor will be splitted to
    SAVED_data_dir : str
        directory with other sub-directories for storing f-m returns of specific granularity and asset universe
        
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

    """

    ### factor mimicking returns pro vsechny faktory (pro vsechny assety) - pocitani trva dlouho => nastavit nepocitani znova pri kazde rebalanci 
    # nacist pripadne existujici a dopocitat zbytek - f-m-r jsou urceny hlavne seznamem features a assetu a frekvenci rebalance 
    # seznam assetu se predpoklada zatim nemenny/dany (vypocet je ovlivnen seznam assetu, ten ale neovlivnuje vystupni format)
    # => tj. v aplikaci se musi ulozene f-m returns propojit na seznam assetu a frekvenci rebalance (seznam features se zjisti z nactenych dat)
    path_port_fmret_diff_cum = os.path.join(SAVED_data_dir, "port_fmret_diff_cum.parquet")
    path_port_fmret_cum = os.path.join(SAVED_data_dir, "port_fmret_cum.parquet")
    path_port_fmcorr = os.path.join(SAVED_data_dir, "port_fmcorr.parquet")
    path_port_fmret_diff = os.path.join(SAVED_data_dir, "port_fmret_diff.parquet") # potreba jen pro pomocny vypocet
    path_port_fmret = os.path.join(SAVED_data_dir, "port_fmret.parquet") # potreba jen pro pomocny vypocet
    if os.path.exists(path_port_fmret_diff_cum) and os.path.exists(path_port_fmret_cum) and os.path.exists(path_port_fmcorr) and \
        os.path.exists(path_port_fmret_diff) and os.path.exists(path_port_fmret):
        print("F-m returns - reading precomputed ...")
        
        port_fmret_diff_cum_tmp = pd.read_parquet(path_port_fmret_diff_cum.replace(".csv", ".parquet")) # .csv se nema kde najit/nahradit
        port_fmret_cum_tmp = pd.read_parquet(path_port_fmret_cum.replace(".csv", ".parquet"))
        port_fmcorr_tmp = pd.read_parquet(path_port_fmcorr.replace(".csv", ".parquet"))
        port_fmret_diff_tmp = pd.read_parquet(path_port_fmret_diff.replace(".csv", ".parquet"))
        port_fmret_tmp = pd.read_parquet(path_port_fmret.replace(".csv", ".parquet"))
        # z nactenych stringovych nazvu kvantilu prevest na ciselne - jinak problem pri concatu a ulozeni .parquet souboru
        port_fmret_cum_tmp.columns = port_fmret_cum_tmp.columns.set_levels(levels=port_fmret_cum_tmp.columns.levels[1].astype(int), level=1) 
        port_fmret_tmp.columns = port_fmret_tmp.columns.set_levels(levels=port_fmret_tmp.columns.levels[1].astype(int), level=1) 
        
        # zjistit, u jakych features pripadne chybi "nejnovejsi" f-m returns (je to na in-samplu -> nemelo by chybet nic) 
        # .. f-m returns shiftuji faktor o +1 -> pri dopocitani vzit o 1 pozorovani do historie vice, aby k prvnimu "chybejicimu" returnu byly faktory        
        features_keep = pd.unique( pd.DataFrame(data=list(features_in.columns))[1] )
        fm_features_incomplete = list(set(features_keep).intersection( set(list(port_fmcorr_tmp.columns)) )) # potencialne nekompletni
        features_in_new = features_in.loc[port_fmcorr_tmp.index[-1]:, (slice(None), fm_features_incomplete)] # zacatek novych o 1 drive nez 1. chybejici
        target_in_new = target_in.loc[port_fmcorr_tmp.index[-1]:, :] # totez u targetu
        if len(features_in_new)>1: # kdyz neco chybelo: delka jen 1 = neni nic noveho (zacatek novych o 1 drive nez 1. chybejici - kvuli zpusobu vypoctu f-m-r)
            print("F-m returns - computing for missing observations ...")
            port_fmret_diff_new, _, port_fmret_new, _, port_fmcorr_new, _ \
                = fs.all_factors_mimicking_returns(features_in_new, target_in_new, nQuant, verbose=True) # *cum* se museji dopocitat zvlast - po priconcateni
            
            port_fmret_diff_all = pd.concat(objs=[port_fmret_diff_tmp.loc[:, fm_features_incomplete], # jen pozadovane feat. (v pripade ulozeni f-m-r pro vice feat.)
                                                  port_fmret_diff_new.loc[port_fmret_diff_new.index[1]:]], axis="index") # od 2. pozorovani
            port_fmcorr_all = pd.concat(objs=[port_fmcorr_tmp.loc[:, fm_features_incomplete], # jen pozadovane features (kdyby bylo ulozeno f-m-r pro vice features) 
                                              port_fmcorr_new.loc[port_fmcorr_new.index[1]:]], axis="index") # od 2. pozorovani
            # multilevelovy concat v indexu - pozor na stejne datove typy nazvu v levelech 
            port_fmret_all = pd.concat(objs=[port_fmret_tmp.loc[:, (fm_features_incomplete, slice(None)) ], # jen pozadovane feat. (v pripade ulozeni f-m-r pro vice feat.)  
                                             port_fmret_new.loc[port_fmret_new.index[1]:] ], axis="index") # od 2. pozorovani
        else: # kdyz nic nechybelo - zustane se to nactene
            print("F-m returns - no new observations: using only stored ones ...")
            port_fmret_all = port_fmret_tmp.loc[:, (fm_features_incomplete, slice(None))] 
            port_fmret_diff_all = port_fmret_diff_tmp.loc[:, fm_features_incomplete]
            port_fmcorr_all = port_fmcorr_tmp.loc[:, fm_features_incomplete]
            
        # zjistit jake features chybi uplne
        fm_features_missing = set(features_keep) - set(list(port_fmcorr_tmp.columns)) 
        features_in_mis = pd.DataFrame()
        if len(fm_features_missing)>0: # kdyz nejake features chybely
            features_in_mis = features_in.loc[:, (slice(None), fm_features_missing)]
            print("F-m returns - computing for whole missing features ...")
            port_fmret_diff_new, _, port_fmret_new, _, port_fmcorr_new, _ \
                = fs.all_factors_mimicking_returns(features_in_mis, target_in, nQuant, verbose=True) # *cum* se museji dopocitat zvlast - po priconcateni
        else: # kdyz nic nechybelo - zustanou prazdny df
            print("F-m returns - no features missing completely (all observations) ...")
            port_fmret_new = pd.DataFrame()
            port_fmret_diff_new = pd.DataFrame()
            port_fmcorr_new = pd.DataFrame()
    
        # spojit dopocitane casti
        port_fmret = pd.concat(objs=[port_fmret_all, port_fmret_new], axis="columns").sort_index(axis="columns")
        port_fmret_diff = pd.concat(objs=[port_fmret_diff_all, port_fmret_diff_new], axis="columns").sort_index(axis="columns")
        port_fmcorr = pd.concat(objs=[port_fmcorr_all, port_fmcorr_new], axis="columns").sort_index(axis="columns")
        
        # dopocitat *cum* (kumulativni kvantilove prumery, rozdily nejlepsiho a nejhorsiho kvantilu) 
        port_fmret_cum = port_fmret.cumsum(axis=0)
        port_fmret_diff_cum = port_fmret_diff.cumsum(axis=0)
    
    else: # kompletne spocitat factor-mimicking data
        os.makedirs(SAVED_data_dir, exist_ok=True) # nevyhodi vyjimku kdyz slozka existuje
        
        print("F-m returns - computing all features completely ...")
        port_fmret_diff, port_fmret_diff_cum, port_fmret, port_fmret_cum, port_fmcorr, port_nrand \
            = fs.all_factors_mimicking_returns(features_in, target_in, nQuant, verbose=True)
        #_, port_fmret_diff_cum, _, port_fmret_cum, port_fmcorr, _ = fs.all_factors_mimicking_returns(features_in, target_in, nQuant, verbose=fm_verbose)
        
        """
        # TEST - ulozit, aby neco chybelo
        port_fmret_diff_cum.iloc[:-3].drop(labels=["Skewness_22", "SUV_Coef_15"], axis="columns").to_csv(path_port_fmret_diff_cum)
        port_fmret_cum.iloc[:-3].drop(labels=["Skewness_22", "SUV_Coef_15"], axis="columns").to_csv(path_port_fmret_cum)
        port_fmcorr.iloc[:-3].drop(labels=["Skewness_22", "SUV_Coef_15"], axis="columns").to_csv(path_port_fmcorr)
        
        port_fmret_diff.iloc[:-3].drop(labels=["Skewness_22", "SUV_Coef_15"], axis="columns").to_csv(path_port_fmret_diff)
        port_fmret.iloc[:-3].drop(labels=["Skewness_22", "SUV_Coef_15"], axis="columns").to_csv(path_port_fmret)
        """
    
    # ... a ulozit - prepsat aktualnimi kompletnimi daty
    # ValueError: parquet must have string column names for all values in each level of the MultiIndex
    port_fmret.columns = port_fmret.columns.set_levels(levels=port_fmret.columns.levels[1].astype(str), level=1) 
    port_fmret_cum.columns = port_fmret_cum.columns.set_levels(levels=port_fmret_cum.columns.levels[1].astype(str), level=1) 
    port_fmret_diff_cum.to_parquet(path_port_fmret_diff_cum)
    port_fmret_cum.to_parquet(path_port_fmret_cum)
    port_fmcorr.to_parquet(path_port_fmcorr)
    port_fmret_diff.to_parquet(path_port_fmret_diff)
    port_fmret.to_parquet(path_port_fmret)

    return port_fmret_diff, port_fmret_diff_cum, port_fmret, port_fmret_cum, port_fmcorr










