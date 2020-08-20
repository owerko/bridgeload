import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime as dt
import pandas as pd

date = []
Rbin23 = []
Rbin39 = []
datetc = []
tc23 = []
tc39 = []

with open('ibis.txt', 'r') as f:
    content = f.readlines()
    for x in content:
        row = x.split()
        date.append((row[0]))
        Rbin23.append(float(row[1]))
        Rbin39.append(float(row[2]))

with open('TC.txt', 'r') as f:
    content = f.readlines()
    for x in content:
        row = x.split()
        datetc.append((row[0]))
        tc39.append(float(row[1]))
        tc23.append(float(row[2]))

print(datetc)
print(tc23)

time = np.asarray(date)
y = np.asarray(Rbin39)

X = []

for index, value in enumerate(y):
    X.append(index)

X = np.asarray(X)
X = X.reshape(-1,1)
model = LinearRegression().fit(X,y)
r2 = model.score(X, y)
print(f'Intercept: {model.intercept_}, Slope: {model.coef_}, E2: {r2}')
Y_pred = model.predict(X)

# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


fig, ax = plt.subplots()
plt.plot(date, Rbin39)
plt.plot(date, Y_pred)

plt.scatter(datetc, tc39, marker='o', color='b', edgecolors='r', alpha=0.5, label="TC")
# plt.plot(date, Rbin39)
plt.xticks(rotation=90)
plt.xlabel('Time [hh:mm:ss]')
plt.ylabel('Deformation [mm]')
ax.xaxis.set_major_locator(ticker.LinearLocator(10))
plt.show()
