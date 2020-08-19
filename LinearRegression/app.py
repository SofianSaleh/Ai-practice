import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

plt.style.use('ggplot')
df = quandl.get('WIKI/GOOGL')
# selecting columns
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
# creating relation  columns
df['HL_PCT'] = (df["Adj. High"] - df["Adj. Close"]) / \
    df["Adj. Close"] * 100.0
df['PCT_CHANGE'] = (df["Adj. Close"] - df["Adj. Open"]) / \
    df["Adj. Open"] * 100.0


df = df[["Adj. Close", "HL_PCT", "PCT_CHANGE", "Adj. Volume"]]
# defining the labels
forecast_col = "Adj. Close"
# filling the null values because we cannot work with null values
df.fillna(-99999, inplace=True)
# defining the output and using 1% of the data to predict and how many days in advanced
forecast_out = int(math.ceil(0.1 * len(df)))
print(forecast_out)
# siffting the column
df['label'] = df[forecast_col].shift(-forecast_out)

# defining labels and features X = features y = labels

X = np.array(df.drop(['label'], 1))
# scaling the values to match
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
# data to predict that has no coresponding y
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

# training data takes X,y shuffles them up to increase accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define a classifier to classify data
clf = LinearRegression(n_jobs=-1)
# fit the results and see the outcome
clf.fit(X_train, y_train)

# saving the trained data using pickle
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)
# train data on seprate data so we get more accuracy
accuracy = clf.score(X_test, y_test)

# predicting values

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df["Forecast"] = np.nan


last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
