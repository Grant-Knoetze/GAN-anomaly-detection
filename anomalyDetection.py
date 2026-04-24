import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. Synthetic Data Generation ---
# We simulate network traffic features (e.g., Packet Size, Duration, Request Frequency)
def generate_data(n_samples=1000, is_anomaly=False):
    if not is_anomaly:
        # Normal traffic: Centered around a specific "benign" distribution
        data = np.random.normal(loc=0.5, scale=0.1, size=(n_samples, 5))
    else:
        # Anomalous traffic: Outliers that deviate significantly
        data = np.random.normal(loc=1.5, scale=0.3, size=(n_samples, 5))
    return data.astype(np.float32)

# Prepare training data (Normal only) and testing data (Normal + Malicious)
train_data_raw = generate_data(n_samples=2000, is_anomaly=False)
test_normal_raw = generate_data(n_samples=500, is_anomaly=False)
test_attack_raw = generate_data(n_samples=500, is_anomaly=True)

# Normalize data to [0, 1] range
scaler = MinMaxScaler()
train_data = torch.FloatTensor(scaler.fit_transform(train_data_raw))
test_normal = torch.FloatTensor(scaler.transform(test_normal_raw))
test_attack = torch.FloatTensor(scaler.transform(test_attack_raw))

# --- 2. Defining the Autoencoder ---
class SecurityAutoencoder(nn.Module):
    def __init__(self):
        super(SecurityAutoencoder, self).__init__()
        # Encoder: Compresses 5 features down to 2
        self.encoder = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 2) # Bottleneck
        )
        # Decoder: Attempts to reconstruct original 5 features
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 3. Training on "Normal" Data Oanly ---a
model = SecurityAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Training Autoencoder on 'Normal' behavior baseline...")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_data)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.6f}")

# --- 4. Anomaly Detection via Reconstruction Error ---
model.eval()
with torch.no_grad():
    # Calculate error for normal test data
    recon_normal = model(test_normal)
    error_normal = torch.mean((test_normal - recon_normal)**2, dim=1).numpy()

    # Calculate error for malicious test data
    recon_attack = model(test_attack)
    error_attack = torch.mean((test_attack - recon_attack)**2, dim=1).numpy()

# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
plt.hist(error_normal, bins=50, alpha=0.6, label='Normal Traffic (Low Error)', color='blue')
plt.hist(error_attack, bins=50, alpha=0.6, label='Malicious Traffic (High Error)', color='red')
plt.axvline(x=0.02, color='black', linestyle='--', label='Detection Threshold') # Arbitrary threshold

plt.title('Anomaly Detection: Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error (MSE)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print("\nDemonstration Complete.")
print("Note how the 'Normal' traffic has low error because the model knows how to compress it.")
print("The 'Malicious' traffic has significantly higher error, making it detectable as an anomaly.")
