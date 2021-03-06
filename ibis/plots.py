import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.linear_model import LinearRegression
from statistics import mean, stdev

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

print(tc23)

time = np.asarray(date)
y = np.asarray(Rbin39)
y23 = np.asarray(Rbin23)

X = []

for index, value in enumerate(y):
    X.append(index)

X = np.asarray(X)
X = X.reshape(-1, 1)
model = LinearRegression().fit(X, y)
r2 = model.score(X, y)
print(f'Intercept: {model.intercept_}, Slope: {model.coef_}, E2: {r2}')
Y_pred = model.predict(X)

model23 = LinearRegression().fit(X, y23)
r2 = model23.score(X, y23)
print(f'Intercept: {model23.intercept_}, Slope: {model23.coef_}, E2: {r2}')
Y_pred23 = model23.predict(X)

timetc = np.asarray(datetc)
ytc = np.asarray(tc39)
ytc23 = np.asarray(tc23)

Xtc = []
for i, v in enumerate(timetc):
    Xtc.append(i)

Xtc = np.asarray(Xtc)
Xtc = Xtc.reshape(-1, 1)

modeltc39 = LinearRegression().fit(Xtc, ytc)
r2 = modeltc39.score(Xtc, ytc)
print(f'Intercept: {modeltc39.intercept_}, Slope: {modeltc39.coef_}, E2: {r2}')
Y_pred_tc39 = modeltc39.predict(Xtc)

modeltc23 = LinearRegression().fit(Xtc, ytc23)
r2 = modeltc23.score(Xtc, ytc23)
print(f'Intercept: {modeltc23.intercept_}, Slope: {modeltc23.coef_}, E2: {r2}')
Y_pred_tc23 = modeltc23.predict(Xtc)

# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

print(ytc)
print(model.predict(X))
print(date)
print(datetc)

indeksy = []
for i in range(len(datetc)):
    for j in range(len(date)):
        if datetc[i] == date[j]:
            indeksy.append(j)

print(indeksy)

dealta39 = []
for i in range(len(indeksy)):
    d = tc39[i] - model.predict(X)[indeksy[i]]
    dealta39.append(d)
print(dealta39)

dealta23 = []
for i in range(len(indeksy)):
    d = tc23[i] - model23.predict(X)[indeksy[i]]
    dealta23.append(d)
print(dealta23)

print(mean(dealta23), stdev(dealta23))
print(mean(dealta39), stdev(dealta39))

# plt.figure(1)
# fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
# plt.subplots_adjust(bottom=0.3)
# plt.plot(date, Rbin39, 'k', label='IBIS data')
# plt.plot(date, Y_pred, 'm-.', label='IBIS linear regression')
# plt.plot(datetc, tc39, 'ro', label='TC')
# # plt.plot(datetc, Y_pred_tc39, 'g-', label='TC linear regression')
# # plt.plot(date, Rbin39)
# plt.xticks(rotation=90)
# plt.xlabel('Time [hh:mm:ss]')
# plt.ylabel('Deformation [mm]')
# plt.title('Rbin 39 observed with IBIS and TC')
# ax.xaxis.set_major_locator(ticker.LinearLocator(10))
# plt.legend()
# plt.show()
#
# fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
# plt.subplots_adjust(bottom=0.34)
# plt.plot(date, Rbin23, 'k', label='IBIS data')
# plt.plot(date, Y_pred23, 'm-.', label='IBIS linear regression')
# plt.plot(datetc, tc23, 'ro', label='TC')
# # plt.plot(datetc, Y_pred_tc23, 'g-', label='TC linear regression')
# # plt.scatter(datetc, tc23, marker='o', color='g', edgecolors='black', alpha=0.5, label="TC")
# plt.xticks(rotation=90)
# plt.xlabel('Time [hh:mm:ss]')
# plt.ylabel('Deformation [mm]')
# plt.title('Rbin 23 observed with IBIS and TC')
# ax.xaxis.set_major_locator(ticker.LinearLocator(10))
# plt.legend()
# plt.show()

xx = [1, 2, 3, 4, 5, 6, 7]
print(dealta23)
print(dealta39)
print(xx)

dx = np.asarray(xx).reshape(-1,1)

modeldiff23 = LinearRegression().fit(dx, dealta23)
modeldiff39 = LinearRegression().fit(dx, dealta39)

y_pred_d23 = modeldiff23.predict(dx)
y_pred_d39 = modeldiff39.predict(dx)

r2 = modeldiff23.score(dx, dealta23)
print(f'Intercept: {modeldiff23.intercept_}, Slope: {modeldiff23.coef_}, E2: {r2}')

r2 = modeldiff23.score(dx, dealta39)
print(f'Intercept: {modeldiff39.intercept_}, Slope: {modeldiff39.coef_}, E2: {r2}')


fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
plt.plot(xx, dealta23, '-ok', color='black', label='Rbin23 diff.')
plt.plot(xx, y_pred_d23, 'k:', label='Rbin23 diff. LR')
plt.plot(xx, dealta39, '-or', color='red', label='Rbin39 diff.')
plt.plot(xx, y_pred_d39, 'r:', label='Rbin39 diff. LR')
plt.xlabel('Observation no.')
plt.ylabel('Difference TC - IBIS_LR[mm]')
plt.legend()
plt.show()
