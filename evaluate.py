import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.vae_model import TradingVAE, loss_function
from utils.data_loader import fetch_and_prepare_data
import random

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

SEQ_LEN = 20
INPUT_DIM = 2
HIDDEN_DIM = 64
LATENT_DIM = 16

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test, _, raw_data = fetch_and_prepare_data("^GSPC", "2025-05-01", "2026-05-01", SEQ_LEN, scaler=scaler)

model = TradingVAE(SEQ_LEN, INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
model.load_state_dict(torch.load("models/vae_trading.pth"))
model.eval()

with torch.no_grad():
    recon_data, mu, logvar = model(X_test)
    test_loss = loss_function(recon_data, X_test, mu, logvar)

    z_random = torch.randn(len(X_test), LATENT_DIM)
    generated_data = model.decode(z_random)

print(f"Out-of-Sample Loss (ELBO): {test_loss.item() / len(X_test):.4f}")

real_returns = X_test[:, -1, 0].numpy()
gen_returns = generated_data[:, -1, 0].numpy()

plt.figure(figsize=(10, 6))
sns.kdeplot(real_returns, fill=True, color="green", label="Real Distribution")
sns.kdeplot(gen_returns, fill=True, color="red", label="Generated Distribution")
plt.title("Out-of-Sample VAE Evaluation")
plt.xlabel("Scaled Log Return")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig("vae_distribution.png", dpi=300, bbox_inches='tight')
print("Graph saved successfully as 'vae_distribution.png'")
