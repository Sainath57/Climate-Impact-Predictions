# Import Libraries
from symtable import Class

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller,pacf,acf
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

# Load Data
data = pd.read_csv('covid_19_weekly.csv')
y = data['new_deaths'].dropna().values  # Remove missing values

#print(len(y))

# Remove Outliers using Z-Score method
z_scores = (y - np.mean(y)) / np.std(y)
y_cleaned = y[np.abs(z_scores) < 3]

#print(len(y_cleaned))
y_cleaned_diff = np.diff(y_cleaned)

# Perform Augmented Dickey-Fuller ADF Test
result = adfuller(y_cleaned)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Interpret result
if result[1] < 0.05:
    print("The series is stationary.")
else:
    print("The series is non-stationary.")


#print(acf(y_cleaned))
#print(pacf(y_cleaned))

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(y, label='New Deaths')
plt.title('Time Series of New Deaths')
plt.xlabel('Day')
plt.ylabel('New Deaths')
plt.legend()
plt.grid()
plt.show()

plot_pacf(y_cleaned, title="PACF of Series")
plot_acf(y_cleaned, title="ACF of Series")


plot_pacf(y_cleaned_diff, title="PACF of Cleaned Series")
plot_acf(y_cleaned_diff, title="ACF of Cleaned Series")

plt.show()

# Fit the ARMA(1,3) model
model = ARIMA(y_cleaned, order=(1, 0, 3))  # AR order = 1, differencing = 0, MA order = 3
model_fit = model.fit()

# Print the summary of the model
print(model_fit.summary())

# Plot residuals
residuals = model_fit.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title("Residuals of ARMA(2,3) Model")
plt.show()

# Histogram of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20, edgecolor='k')
plt.title("Error Histogram")
plt.show()

# Ljung-Box test for residual autocorrelation
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)
print("==========================================================")


# Predictions from the model
y_pred = model_fit.fittedvalues  # Fitted values (predicted)
y_actual = y_cleaned  # Actual values

# 1. CSSE: Cumulative Sum of Squared Errors
csse = np.sum((y_actual - y_pred) ** 2)

# 2. SST: Total Sum of Squares
sst = np.sum((y_actual - np.mean(y_actual)) ** 2)

# 3. R^2: Coefficient of Determination
r2 = 1 - (csse / sst)

# 4. MAE: Mean Absolute Error
mae = np.mean(np.abs(y_actual - y_pred))

# 5. SMAPE: Symmetric Mean Absolute Percentage Error
smape = 100 * np.mean(np.abs(y_actual - y_pred) / ((np.abs(y_actual) + np.abs(y_pred)) / 2))

# 6. MSE: Mean Square Error
mse = csse / (len(y_actual)-1)

# Print metrics
print(f"CSSE: {csse}")
print(f"SST: {sst}")
print(f"R^2: {r2}")
print(f"MAE: {mae}")
print(f"SMAPE: {smape}%")
print(f"MSE: {mse}")


##############################
#An Example discussed in Class
##############################
# Import the numpy library
import numpy as np
"""
# Define the dataset
x = np.array([3,4,2,5,7,9,8,6,0])
y = np.array([1,3,4,2,5,7,9,8,6])
def Pearson_correlation(X,Y):
	if len(X)==len(Y):
		Sum_xy = sum((X-X.mean())*(Y-Y.mean()))
		Sum_x_squared = sum((X-X.mean())**2)
		Sum_y_squared = sum((Y-Y.mean())**2)	 
		corr = Sum_xy / np.sqrt(Sum_x_squared * Sum_y_squared)
	return corr
			
print(Pearson_correlation(x,y)) 
#print(Pearson_correlation(x,x))
print(np.corrcoef(x,y))
"""
y = np.array([3,4,2,5,7,9,8,6,0])
zt = np.array([-3.5,-1.5,-0.5,-1.5,0.5,2.5,4.5,3.5,1.5])
phi = 0.4821
yt = np.array([-3.5,-1.5,-0.5,-1.5,0.5,2.5,4.5,3.5,1.5])
yhat = np.array([2.82,3.77,4.25,3.77,4.74,5.70,6.66,6.18,5.22])
e2 = np.array([0.03240000000000006,
               0.0528999999999999,
               5.0625,
               1.5129,
               5.107599999999999,
               10.889999999999999,
               1.7955999999999996,
               0.0323999999999999,
               27.248399999999997])
for i in range(0,len(y)):
    print((y[i]-yhat[i])**2)

print(np.sum(e2))

actual = np.array([1, 3, 4, 2, 5, 7, 9, 8, 6])
predicted = np.array([2.81,3.77,4.25,3.77,4.74,5.70,6.66,6.18,5.22])

# Calculate SMAPE
smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100
print(smape)