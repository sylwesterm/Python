import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

# Load the data
file_path = 'C:\BI\locations.csv'  # Change this to your actual file path
data = pd.read_csv(file_path)

print(data.size)

# Convert DateAndTime to datetime for processing
data['DateAndTime'] = pd.to_datetime(data['DateAndTime'])
data['DateAndTime'] = data['DateAndTime'].dt.floor('T')

# Preparing the data for model training
X = (data['DateAndTime'] - data['DateAndTime'].min()).dt.total_seconds().values.reshape(-1, 1)
y = data['Battery'].values

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict when the battery will be less than 10%
target_battery_level = 10
time_for_target_battery_level = (target_battery_level - model.intercept_) / model.coef_
predicted_datetime_for_target_battery_level = data['DateAndTime'].min() + pd.to_timedelta(time_for_target_battery_level[0], unit='s')

# Generating a range of future timestamps for visualization
predicted_below_10_percent_seconds = (predicted_datetime_for_target_battery_level - data['DateAndTime'].min()).total_seconds()
future_timestamps_seconds_range = np.linspace(X[-1], predicted_below_10_percent_seconds, num=100).reshape(-1, 1)

# Convert seconds back to datetime for plotting
future_dates_range = [data['DateAndTime'].min() + pd.to_timedelta(sec, unit='s') for sec in future_timestamps_seconds_range.flatten()]

# Predict battery levels for this new range
future_battery_levels_range = model.predict(future_timestamps_seconds_range)

print(f'The battery level will fall below {target_battery_level}% at {predicted_datetime_for_target_battery_level}')

# Plotting
fig, ax = plt.subplots(figsize=(12, 7))

# Original battery levels
ax.plot(data['DateAndTime'], data['Battery'], marker='o', linestyle='-', color='blue', label='Actual Battery Level')

# Extended prediction until battery level falls below 10%
ax.plot(future_dates_range, future_battery_levels_range, linestyle='--', color='red', label='Extended Prediction Until <10%')

ax.set_title('Battery Level Over Time with Extended Prediction')
ax.set_xlabel('Date and Time')
ax.set_ylabel('Battery Level (%)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
