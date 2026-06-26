import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import pickle
import os
import random
from models.vae_model import TradingVAE, loss_function
from utils.data_loader import fetch_and_prepare_data

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

SEQ_LEN = 20
INPUT_DIM = 2
HIDDEN_DIM = 64
LATENT_DIM = 16
EPOCHS = 10000
BATCH_SIZE = 64
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train, scaler, _ = fetch_and_prepare_data("^GSPC", "2024-05-01", "2025-05-01", SEQ_LEN)
dataset = TensorDataset(X_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. MOVE MODEL TO GPU ---
model = TradingVAE(SEQ_LEN, INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    MAX_BETA = 0.1
    beta = min(MAX_BETA, (epoch / 25.0) * MAX_BETA)

    for batch in dataloader:
        batch_data = batch[0].to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch_data)
        loss = loss_function(recon_batch, batch_data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {train_loss / len(dataloader.dataset):.4f}')

os.makedirs("models", exist_ok=True)
torch.save(model.cpu().state_dict(), "models/vae_trading.pth")
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training complete. Model and scaler saved")
