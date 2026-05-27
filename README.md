# Ex.No: 6               HOLT WINTERS METHOD
### Date: 19.05.2026



### AIM:To implement Holt-Winters model on Onion Price Data Set and make future predictions

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load CSV dataset
data = pd.read_csv("/content/housing_prices_dataset.csv")

# Remove empty columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Keep only required columns from the new dataset and rename them
# Assuming the price data is in the last column of the DataFrame
last_column_name = data.columns[-1]
data = data[['Year_Built', last_column_name]]
data = data.rename(columns={'Year_Built': 'Year Built', last_column_name: 'Price'})

# Convert 'Year Built' column to datetime
# Fix: Added errors='coerce' to handle invalid dates like '31-04-2027'
# Assuming 'Year Built' contains integer years, convert to string first for proper parsing
data['Year Built'] = pd.to_datetime(data['Year Built'].astype(str), format='%Y', errors='coerce')

# Set 'Year Built' as index
data.set_index('Year Built', inplace=True)

# Display dataset
print(data.head())

# Resample monthly
data_monthly = data.resample('MS').mean()

# Fix: Interpolate missing values after resampling
data_monthly = data_monthly.interpolate()

# Plot original data
data_monthly.plot(title='Monthly Price Data')
plt.xlabel("Year Built")
plt.ylabel("Price")
plt.show()

# Scale the data
scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(
        data_monthly.values.reshape(-1, 1)
    ).flatten(),
    index=data_monthly.index
)

# Plot scaled data
scaled_data.plot(title='Scaled Price Data')
plt.show()

# Add 1 because multiplicative seasonality
# cannot handle non-positive values
scaled_data = scaled_data + 1

# Seasonal decomposition
decomposition = seasonal_decompose(
    data_monthly,
    model='additive',
    period=12
)

decomposition.plot()
plt.show()

# Split train and test data
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Build Holt-Winters model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Forecast test data
test_predictions_add = model_add.forecast(
    steps=len(test_data)
)

# Plot train, test and predictions
ax = train_data.plot(figsize=(10, 5))
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend([
    "train_data",
    "test_predictions_add",
    "test_data"
])

ax.set_title('Visual Evaluation')
plt.show()

# RMSE calculation
rmse = np.sqrt(
    mean_squared_error(
        test_data,
        test_predictions_add
    )
)

print("RMSE:", rmse)

# Standard deviation and mean
print(
    "Standard Deviation:",
    np.sqrt(scaled_data.var())
)

print(
    "Mean:",
    scaled_data.mean()
)

# Final model using complete data
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Predict future values
final_predictions = final_model.forecast(steps=12)

# Plot final prediction
ax = scaled_data.plot(figsize=(10, 5))
final_predictions.plot(ax=ax)

ax.legend([
    "Original Data",
    "Future Predictions"
])

ax.set_xlabel('Year Built')
ax.set_ylabel('Price')
ax.set_title('Price Forecast using Holt-Winters Method')

plt.show()
```

### OUTPUT:
<img width="687" height="650" alt="image" src="https://github.com/user-attachments/assets/60148fbf-2e27-4cd1-b16a-6fa1b5aba723" />

<img width="628" height="515" alt="image" src="https://github.com/user-attachments/assets/74410126-519e-41b6-be71-a9926b69b945" />

<img width="704" height="532" alt="image" src="https://github.com/user-attachments/assets/542dfdcd-697f-4342-9139-af5905a95891" />

<img width="934" height="589" alt="image" src="https://github.com/user-attachments/assets/21ffa635-8c26-4a12-826d-2a4d7493a6ff" />

<img width="956" height="529" alt="image" src="https://github.com/user-attachments/assets/c08bf7a1-38c9-4ab8-bd47-412fd8a64e51" />




### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
