import yahooquery as yq
import pandas as pd
import os
import data_storage as ds
import yfinance as yf
import logging

#%%
logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
#%%
##############################################################################
#
#                   FUNDAMENTAL DATA - GLOBAL
#
##############################################################################

#############################################################################
#                   FUNDAMENTAL DATA PREPROCESSING
#
# 1) read "messy" data from ../SP-data/fundamental_data/firm_specific/feed_fundamental
# 2) parse desired series
# 3) save to ../SP-data/fundamental_data/firm_specific/src_fundamental
#############################################################################

# Read "messy" Data
class FundamentalProcessor:
    '''
    Processing desired series (whole series, without "last value updating") 
    ---------------------------------------------------------------------------
    # TODO:update last values
    ---------------------------------------------------------------------------
        if data_to_add.empty: # no parquet file and no downloaded data
        pass
    elif os.path.isfile(f"{path_to_files}/{ticker}_daily.parquet"): # check if parquet file exists
        file = pd.read_parquet(f"{path_to_files}/{ticker}_daily.parquet") # load parquet file with old data
        if file.index.max() == today:  # check if index with current date exists
            file.drop(today, inplace=True)  # delete whole row with current date and replece it with new data
        data_to_add = pd.concat([data_to_add, file], axis=0)
        logging.info("Saving to parquet")
        save_file(ticker, data_to_add)

    else:
        logging.info("New file was created") # new downloaded data but no paquet file with old data
        save_file(ticker, data_to_add)
    ---------------------------------------------------------------------------
    '''
    
    def __init__(self, tickers: list):
        self.tickers = tickers
        #self.fundamental_path = self.src_fundamental_path(self.ticker)
        #self.datafeed = self.fundamental_datafeed(ticker)
        #self.fundamental_src = self.parse_desired_fundamental(self.tickers)
        
    def src_fundamental_path(self, ticker):
        
        cwd = os.getcwd()
        SAVED_fundamental = "../SP-data/fundamental_data/firm_specific/src_fundamental/" #{}_daily.parquet".format(ticker)
        DIR_fundamental = os.path.join(cwd, SAVED_fundamental)
        
        if (not os.path.exists(DIR_fundamental)):
            os.makedirs(DIR_fundamental)
            #print(SAVED_fundamental, " - folder for fundamental data: created")
            logging.info("%s - folder for fundamental data: created", SAVED_fundamental)
        else:
            #print("TICKER:", ticker)
            logging.info(" TICKER: %s", ticker)
            #print("Fundamental Data Storage", SAVED_fundamental)
            logging.info("Fundamental Data Storage: %s", SAVED_fundamental)
            
        fundamental_path = DIR_fundamental + "{}_daily.parquet".format(ticker)
        
        return fundamental_path 
        
    def get_fundamental_datafeed(self, ticker):
    
        path_to_feed = "../SP-data/fundamental_data/firm_specific/feed_fundamental/"
        fundamental_pathfeed = path_to_feed + "{}_daily.parquet".format(ticker)
        datafeed = None
        try:
            datafeed = pd.read_parquet(fundamental_pathfeed)
        except:
            logging.info(" FILE DOES NOT EXIST: %s", fundamental_pathfeed)
    
        if datafeed is not None:
            logging.info("%s: READING FROM SOURCE...", ticker)
            return datafeed 
        else:
            return
    
    def get_operatingCF(self, fundamental_datafeed):

        cf = fundamental_datafeed.loc[:, (slice(None), ["operatingCashflow", "operatingCashFlow"])].droplevel(level=0, axis="columns")
        cf1 = cf.loc[:, "operatingCashflow"].dropna()
        cf2 = cf.loc[:, "operatingCashFlow"]
        operatingCF = cf1.combine_first(cf2)
        operatingCF = operatingCF[~operatingCF.index.duplicated(keep="first")] # some dates are duplicated, one keeping value while the second is Nan
        operatingCF = pd.DataFrame(operatingCF) # because of mMltiIndex
        #operatingCF = fill_missing_values(operatingCF, strategy="median")
        
        return operatingCF
    
    def get_dividendPerShare(self, ticker, fundamental_datafeed):
        
        # Price Data Gathering
        price = self.get_price(ticker)
        # DividendPerShare
        dividendYield = fundamental_datafeed.droplevel(level=0, axis="columns").loc[:, "dividendYield"].to_frame()
        dividendYield = dividendYield[~dividendYield.index.duplicated(keep="first")]
        
        dividendPerShare = dividendYield.mul(price, axis="index") # dividendPerShare = divYield * Market Value per Share, i.e. Price
        dividendPerShare.columns = ["dividendPerShare"]
        
        return dividendPerShare
    
    def get_earningsPerShare(self, ticker, fundamental_datafeed):    
        "in behalf of improver"
        earningsPerShare = fundamental_datafeed.droplevel(level=0, axis="columns").loc[:, "earningsPerShare"].to_frame()
        duplicated = earningsPerShare[earningsPerShare.index.duplicated(keep=False)].dropna()
        earningsPerShare = earningsPerShare.combine_first(duplicated)
        earningsPerShare = earningsPerShare[~earningsPerShare.index.duplicated(keep="first")]
        
        return earningsPerShare
    
    def get_marketCap(self, fundamental_datafeed):
        "in behalf of improver"
        marketCap = fundamental_datafeed.droplevel(level=0, axis="columns").loc[:, "marketCap"].to_frame()
        duplicated = marketCap[marketCap.index.duplicated(keep=False)].dropna()
        marketCap = marketCap.combine_first(duplicated)
        marketCap = marketCap[~marketCap.index.duplicated(keep="first")]
    
        return marketCap

    def get_revenuePerShare(self, fundamental_datafeed):
        "Same as SalesPerShare"
        "in behalf of improver"
        revenuePerShare = fundamental_datafeed.droplevel(level=0, axis="columns").loc[:, "revenuePerShare"].to_frame()
        duplicated = revenuePerShare[revenuePerShare.index.duplicated(keep=False)].dropna()
        revenuePerShare = revenuePerShare.combine_first(duplicated)
        revenuePerShare = revenuePerShare[~revenuePerShare.index.duplicated(keep="first")]
    
        return revenuePerShare
    
    def get_priceToBook(self, fundamental_datafeed):
        "in behalf of improver"
        priceToBook = fundamental_datafeed.droplevel(level=0, axis="columns").loc[:, "priceToBook"].to_frame()
        duplicated = priceToBook[priceToBook.index.duplicated(keep=False)].dropna()
        priceToBook = priceToBook.combine_first(duplicated)
        priceToBook = priceToBook[~priceToBook.index.duplicated(keep="first")]
    
        return priceToBook

    def get_priceCashFlow(self, ticker, fundamental_datafeed):
        
        # Price Data Gathering
        price = self.get_price(ticker)
        # Price Cash Flow
        cashFlowPerShare = fundamental_datafeed.droplevel(level=0, axis="columns").loc[:, "cashFlowPerShare"].to_frame()
        cashFlowPerShare = cashFlowPerShare[~cashFlowPerShare.index.duplicated(keep="first")]
        
        priceCashFlow = price.mul(price, axis="index").to_frame() # dividendPerShare = divYield * Market Value per Share, i.e. Price
        priceCashFlow.columns = ["priceCashFlow"]
        
        return priceCashFlow
    
    def get_grossMargins(self, fundamental_datafeed):
        "in behalf of improver"
        grossMargins = fundamental_datafeed.droplevel(level=0, axis="columns").loc[:, "grossMargins"].to_frame()
        duplicated = grossMargins[grossMargins.index.duplicated(keep=False)].dropna()
        grossMargins = grossMargins.combine_first(duplicated)
        grossMargins = grossMargins[~grossMargins.index.duplicated(keep="first")]
        
        return grossMargins
    
    def get_priceEarnings(self, ticker, fundamental_datafeed):
        
        earningsPerShare = self.get_earningsPerShare(ticker, fundamental_datafeed)
        price = self.get_price(ticker).to_frame()
        priceEarnings = price.divide(earningsPerShare).drop("Close", axis="columns")
        priceEarnings.columns = ["priceEarnings"]
        
        return priceEarnings
    
    def get_price(self, ticker):
        
        # Price Data Gathering
        path_to_feed = "../SP-data/price_data/{}-f1.parquet".format(ticker)
        #print("START PRICE DATA GATHERING...")
        if os.path.isfile(path_to_feed):
            #print("READING...", path_to_feed)
            price = pd.read_parquet(path_to_feed)
            price = price.loc[:, "Close"]
        else:
            #print("FILE DOES NOT EXIST", path_to_feed)
            #print("QUERYING YAHOO DATA...")
            yf_ticker = yf.Ticker(ticker)
            yf_df = yf_ticker.history(period="1d", start="2010-5-31", auto_adjust=True)
            #print(ticker, "PRICE DATA RECEIVED")
            price = yf_df.loc[:, "Close"]
        
        return price
    
    def save_desired_fundamental(self):
        
        for ticker in self.tickers:
            # In-memory Storage
            fundamental_src = pd.DataFrame()
            # Source Data
            fundamental_datafeed = None
            # Datafeed for Specific Ticker
            try:
                fundamental_datafeed = self.get_fundamental_datafeed(ticker)
            except:
                pass
            if fundamental_datafeed is not None:
                try:
                    # OPERATING CASH-FLOW
                    operatingCF = self.get_operatingCF(fundamental_datafeed=fundamental_datafeed)
                    # DIVIDEND PER SHARE
                    dividendPerShare = self.get_dividendPerShare(ticker, fundamental_datafeed)
                    # EARNINGS PER SHARE
                    earningsPerShare = self.get_earningsPerShare(ticker, fundamental_datafeed)
                    #MARKET CAPITALIZATION
                    marketCap = self.get_marketCap(fundamental_datafeed)
                    # REVENUES OER SHARE
                    revenuePerShare = self.get_revenuePerShare(fundamental_datafeed)
                    # PRICE TO BOOK
                    priceToBook = self.get_priceToBook(fundamental_datafeed)
                    # PRICE CASH FLOW
                    priceCashFlow = self.get_priceCashFlow(ticker, fundamental_datafeed)
                    # GROSS MARGINS
                    grossMargins = self.get_grossMargins(fundamental_datafeed)
                    #PRICE EARNINGS
                    priceEarnings = self.get_priceEarnings(ticker, fundamental_datafeed)
                    
                    # Join and save Ticker specific Fundamental Data
                    fundamental_src = pd.concat(objs=[fundamental_src, 
                                                      operatingCF, 
                                                      dividendPerShare, 
                                                      earningsPerShare,
                                                      marketCap,
                                                      revenuePerShare,
                                                      priceToBook,
                                                      priceCashFlow,
                                                      grossMargins,
                                                      priceEarnings
                                                      ],
                                                axis="columns")
                    path_to_save = self.src_fundamental_path(ticker)
                    fundamental_src.to_parquet(path_to_save)
                    logging.info("**********************************************************")
                    logging.info(" %s: FUNDAMENTAL DATA REDESIGNED & SAVED", ticker)
                    logging.info("**********************************************************")
                except KeyError as e:
                    logging.exception(e)
#%%
###############################################################################
# ZLEPSOVAK - IN AN ADEQUATE TIME, NOT SOONER
###############################################################################

#%%
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
###############################################################################
# MISSING VALUES IMPUTATION
###############################################################################
def fill_missing_values(serie_to_fill, strategy="median"):
    """
    Reference
    ---------
        Python Feature Engineering Cookbook, Soledad Galli, 
        Chap.2, Performing multivariate imputation by chained equations

    Parameters
    ----------
    serie_to_fill : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    ##################################################################
    #
    # DATA PROCESSING - Faking Multivariate Imputing
    #
    #################################################################
    # X: original data
    X = pd.DataFrame(data=serie_to_fill) # retype to have pd.Dataframe
    # X_train: preprocessed data to next imputing
    X_train = pd.DataFrame() # training data object
    #train_size = int(len(X)*0.5) # integer value as indexer
    X_train[X.columns+"_1"] = X # containing missing data injection, fake due to be like MICE (Multivariate Imputation by Chained Equations)
    X_train[X.columns+"_2"] = X # containing missing data injection, fake due to be like MICE (Multivariate Imputation by Chained Equations)
    ##################################################################
    #
    # MISSING VALUES IMPUTATION
    #
    #################################################################
    if strategy not in ["median", "bayes", "knn", "nonLin", "forest", "diagnose_all"]:
        return print(">STRATEGY< NOT SUPPORTED, CHECK FUNCTION DOCUMENTATION FOR APPROPRIATE STRATEGY VALUES")
    # Nan Values Imputation - Simple Imputing
    if strategy=="median":
        imputer = SimpleImputer(strategy="median")
    # Nan Values Imputation 
    elif strategy == "bayes":
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
    elif strategy == "knn":
        imputer = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5), max_iter=10, random_state=0)
    elif strategy == "nonLin":
        imputer = IterativeImputer(estimator=DecisionTreeRegressor(max_features='sqrt', random_state=0), max_iter=10, random_state=0)
    elif strategy == "forest":
        imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), max_iter=10, random_state=0)

    if strategy in ["median", "bayes", "knn", "nonLin", "forest"]:
        imputer.fit(X_train)
        X_train_imputed = imputer.transform(X_train)
        imputed = pd.DataFrame(X_train_imputed[:, 0], index=X.index, columns=X.columns) # [:, 0] because we need only 1 column/serie
       
        return imputed
    
    if strategy == "diagnose_all":
        imputer_median = SimpleImputer(strategy="median")    
        imputer_bayes = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
        imputer_knn = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5), max_iter=10, random_state=0)
        imputer_nonLin = IterativeImputer(estimator=DecisionTreeRegressor(max_features='sqrt', random_state=0), max_iter=10, random_state=0)
        imputer_missForest = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=0), max_iter=10, random_state=0)
        
        imputer_median.fit(X_train)
        imputer_bayes.fit(X_train)
        imputer_knn.fit(X_train)
        imputer_nonLin.fit(X_train)
        imputer_missForest.fit(X_train)
        
        X_train_median = imputer_median.transform(X_train)
        X_train_bayes = imputer_bayes.transform(X_train)
        X_train_knn = imputer_knn.transform(X_train)
        X_train_nonLin = imputer_nonLin.transform(X_train)
        X_train_missForest = imputer_missForest.transform(X_train)
        
        imputed_median = pd.DataFrame(X_train_median[:, 0], columns=X.columns, index=X.index)
        imputed_bayes = pd.DataFrame(X_train_bayes[:, 0], columns=X.columns, index=X.index)
        imputed_knn = pd.DataFrame(X_train_knn[:, 0], columns=X.columns, index=X.index)
        imputed_nonLin = pd.DataFrame(X_train_nonLin[:, 0], columns=X.columns, index=X.index)
        imputed_missForest = pd.DataFrame(X_train_missForest[:, 0], columns=X.columns, index=X.index)
        
        ##############################################################
        # PLOT SPECIFIC IMPUTATIONS
        ##############################################################
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        X.plot(kind="kde", ax=ax, color="blue")
        imputed_median.plot(kind="kde", ax=ax, color="green")
        imputed_bayes.plot(kind="kde", ax=ax, color="red")
        imputed_knn.plot(kind="kde", ax=ax, color="black")
        imputed_nonLin.plot(kind="kde", ax=ax, color="orange")
        imputed_missForest.plot(kind="kde", ax=ax, color="yellow")
        # legend
        lines, labels = ax.get_legend_handles_labels()
        labels = ["original", "Median", "Bayes", "KNN", "Trees", "missForest"]
        ax.legend(lines, labels, loc="best")
        plt.show()
        
        return imputed_median, imputed_bayes, imputed_knn, imputed_nonLin, imputed_missForest 

# operatingCF_imputed = fill_missing_values(serie_to_fill=operatingCF, strategy="diagnose_all")
#%%
def main():
    tickers = ds.get_sp100()
    fundamentals = FundamentalProcessor(tickers)
    fundamentals.save_desired_fundamental()

if __name__ == "__main__":
    main()

#EoF
