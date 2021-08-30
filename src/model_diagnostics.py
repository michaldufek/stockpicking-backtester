#%%
###############################################################################
#
#                           DIAGNOSTICS
#
###############################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
#from sklearn import linear_model

import data_computer as dc
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# INVERSE TRANSFORM (DESCALING) of the PREDICTED TARGET
def prediction_descale(df, y_true, y_pred):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    target_all_out : TYPE
        DESCRIPTION.

    Returns
    -------
    predictions_descaled : TYPE
        DESCRIPTION.

    """
    returns = pd.DataFrame(np.log(df.loc[:, (slice(None), "Close")].to_numpy()/df.loc[:, (slice(None), "Close")].shift(1).to_numpy()), index=df.index, columns=df.columns.levels[0])
    returns = dc.compute_return(df)
    returns = returns.iloc[-len(y_true.index):, :]
    # 1. fit scaler od outstanding data
    scaler = StandardScaler()
    # 2. perform inverse_transform on scaled predictions
    scaler = scaler.fit(returns.T)
    # 3. compare outputs
    predictions_descaled = pd.DataFrame(data=scaler.inverse_transform(y_pred.T).T, index=y_pred.index, columns=y_pred.columns)
    
    return predictions_descaled

#prediction_descaled = prediction_descale(df, ytrue=target_all_out, y_pred=target_fcast_out)
#%%
# R2
def print_metrics(model, X_test, y_test):
    '''
    Print out model performance metrics: MSE, R2 and HIT ratio

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    mse = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))    
    print('MEAN SQUARED ERROR: %.2f' % mse)
    #print('Mean squared error: %.2f' % (mean_squared_error(y_true=target_out, y_pred=best_mdl.predict(predictors_out))))
    r2 = model.score(X_test, y_test) * 100
    print('R2 MULTIPLIED WITH 100: %.2f' % r2)
    #print('R2 out-of sample multiplied by 100: %.2f' % (best_mdl.score(predictors_out, target_out)*100))
    # HIT RATIO
    hit_ratio = np.mean(model.predict(X_test) * y_test> 0)
    print('HIT RATIO: %.2f' % hit_ratio)
#%%
def plot_learning_curve(estimator, X_train, y_train, scoring='neg_mean_squared_error'):
    
    train_sizes, train_scores, validation_scores = learning_curve(estimator, X_train, y_train, scoring='neg_mean_squared_error', verbose=10)
    
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    
    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()

#estimator = linear_model.Ridge(fit_intercept=True, normalize=False, solver="auto")
#plot_learning_curve(estimator=estimator, X_train=predictors_in, y_train=target_in, scoring='neg_mean_squared_error')