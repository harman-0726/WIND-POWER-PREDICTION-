import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load dataset
df = pd.read_csv("C://projects//wind_power_project//Location2.csv")

# Drop 'Time' column
df = df.drop(columns=['Time'])

# Check missing values
print(f"Missing value per column: {df.isnull().sum()}")

# Split X and Y
X = df.drop(columns=['Power']).values
Y = df[['Power']].values

# Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# X_train and Y_train
x_train, x_test, y_train, y_test = train_test_split(x_scaled, Y, random_state=42, test_size=0.2)

# Convert to pytorch tensors
X_train = torch.tensor(x_train, dtype=torch.float32)
X_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# Define the ANN Model
class WindPowerANN(nn.Module):
    def __init__(self, input_dim):
        super(WindPowerANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize Model
input_size = X_train.shape[1]
model = WindPowerANN(input_size)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move to device
model.to(device)

# Training Loop
epochs = 100
train_losses = []

print("\n🚀 Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward + Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

# Plot Training Loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('training_loss.png')
print("\n📊 Training loss plot saved as 'training_loss.png'")

# Evaluation Phase
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        predictions.append(outputs.cpu().numpy())
        actuals.append(labels.cpu().numpy())

# Convert to numpy arrays
predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Calculate Metrics
mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f"\n📊 Model Evaluation Results:")
print(f"MSE  : {mse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"R²   : {r2:.4f}")

# Plot Actual vs Predicted Power
plt.figure(figsize=(8, 5))
plt.scatter(actuals, predictions, alpha=0.6, color='blue')
plt.xlabel("Actual Power")
plt.ylabel("Predicted Power")
plt.title("Actual vs Predicted Power Output")
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')
plt.savefig('prediction_plot.png')
print("📊 Prediction plot saved as 'prediction_plot.png'")

# Test prediction with sample data
new_data = np.array([[19, 90, 12.6, 5.9, 9.3, 70, 65, 10.0]])
new_data_scaled = scaler.transform(new_data)
new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    predicted_power = model(new_tensor).cpu().numpy()[0][0]

print(f"\n⚡ Test Prediction - Predicted Power Output: {predicted_power:.4f} kW")

# ✅ SAVE MODEL AND SCALER
print("\n💾 Saving model and scaler...")
torch.save(model.state_dict(), 'wind_power_model.pth')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model saved as 'wind_power_model.pth'")
print("✅ Scaler saved as 'scaler.pkl'")
print("\n🎉 Training complete! You can now run the Streamlit app.")