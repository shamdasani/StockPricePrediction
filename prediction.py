import pandas as pd
import numpy as np
import quandl, math, time, datetime
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'nwE_WyQ_7VLxsXDpgU7H'
df = quandl.get('WIKI/GOOGL')

df=df[['Adj. Close']]
df.fillna(-99999, inplace=True) # 1) what happens if I take this out?

forecast = int(math.ceil(0.01*len(df))) # 1% of stocks lifetime to be forecasted
df['Forecast'] = df['Adj. Close'].shift(-forecast) # 2) figure out correct shifting model

'''
 if this shift is used, 
 [1% of stocks lifetime] (e.g. 33) days ago is the forcasted price for that day.
 this shift is built to learn from the stocks trend.
'''

# X - Features (Adj. Close) | y - Labels (Forecast)
X = np.array(df.drop(['Forecast'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast:] # understand this!
X = X[:-forecast] # understand this too!

df.dropna(inplace=True) #understand Drop NA
y = np.array(df['Forecast'])

# Training - test size?
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Classifier 
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

# Storing + Accessing Training Data
with open('prediction.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('prediction.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast)

# Fix date + time issues
# RIGHT NOW: Data is just shifting back 5 places as seen in forecast variable
'''

df['Forecast'] = np.nan


last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set: 
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
'''

# Plotting on Graph
style.use('ggplot')

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

print(df.tail(35))

