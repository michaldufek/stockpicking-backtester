from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_classification # to fake model
from sklearn.datasets import make_regression # to fake model
from matplotlib import pyplot
import pandas as pd

from sklearn.linear_model import LinearRegression, ElasticNet
#%%
###############################################################################
#
#                               FAKE MODEL, for development only
#
###############################################################################

# Fake Data

# test regression/classification dataset
# define dataset
#X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
columns = ['R', 'A', 'D', 'E', 'K', 'Z', 'A', 'D', 'E', 'K' ]
X = pd.DataFrame(X, columns=columns)
# Model
model = ElasticNet()
# Fit the model
model.fit(X, y)
#%%
###############################################################################
#
#                               FEATURE IMPORTANCE
#
###############################################################################
# Get Feature Importance
importance = model.coef_
# Summarize Feature Importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# Plot Feature Importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

'''
m = SelectFromModel(ElasticNetCV())

m.fit(X, y)
m.transform(X).shape


if __name__=='__main__':
    main()
else:
    pass
'''
# EoF