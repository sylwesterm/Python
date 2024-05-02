import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim

# Load the data
file_path = 'C:\\BI\\locations.csv'  # Adjust the path to where your file is located
data = pd.read_csv(file_path)

# Convert DateAndTime to datetime
data['DateAndTime'] = pd.to_datetime(data['DateAndTime'])

# Preparing the data for model training
X = (data['DateAndTime'] - data['DateAndTime'].min()).dt.total_seconds().values.reshape(-1, 1)
X_min, X_max = X.min(), X.max()
X = (X - X_min) / (X_max - X_min)  # Min-Max normalization
X = torch.tensor(X, dtype=torch.float32)
y = data['Battery'].values
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

data['DateAndTime'] = data['DateAndTime'].dt.floor('T')
print(data['DateAndTime'].values)

# Define a simple linear model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input and one output

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Adjusted learning rate

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

target_battery_level = 10

# Predict when the battery will be less than 10%
with torch.no_grad():
    # Since X was normalized, we simulate a normalized input for the target prediction
    target_time_normalized = (target_battery_level - X_min) / (X_max - X_min)
    predicted_normalized = model(torch.tensor([[target_time_normalized]], dtype=torch.float32))
    
    # Reverse the normalization for the prediction
    predicted_time_seconds = predicted_normalized.item() * (X_max - X_min) + X_min
    predicted_datetime_for_target_battery_level = data['DateAndTime'].min() + pd.to_timedelta(predicted_time_seconds, unit='s')

print("Predicted time for battery level to fall below 10%:", predicted_datetime_for_target_battery_level)

# Since the focus is on correcting the code, you can follow the previous matplotlib plotting approach to visualize the results.
# Plotting the battery discharge
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
plt.plot(data['DateAndTime'], data['Battery'], linestyle='-', marker='o', color='tab:blue', label='Actual Battery Level')

# Marking the predicted point when the battery falls below 10%
plt.axvline(x=predicted_datetime_for_target_battery_level, color='tab:red', linestyle='--', label='Predicted <10% Battery Level')

# Formatting the plot
plt.title('Battery Discharge and Prediction of <10% Level')
plt.xlabel('Date and Time')
plt.ylabel('Battery Level (%)')
plt.grid(True)
plt.legend()

# Improve date formatting on x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # Adjust the locator based on your data range
plt.gcf().autofmt_xdate()  # Auto-format the dates for better readability

plt.show()