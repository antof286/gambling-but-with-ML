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
model.load_state_dict(torch.load("models/vae_trading.pth", weights_only=True))
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

STARTING_CAPITAL = 10000
NUM_SIMULATIONS = 1000
DAYS_FORWARD = SEQ_LEN

with torch.no_grad():
    _, mu_all, _ = model(X_test)
    empirical_std = mu_all.std(dim=0)
    current_market_state = X_test[-1].unsqueeze(0)
    _, mu_today, _ = model(current_market_state)
    z_sim = torch.randn(NUM_SIMULATIONS, LATENT_DIM) * empirical_std + mu_today
    future_paths_scaled = model.decode(z_sim).numpy()

simulated_scaled_returns = future_paths_scaled[:, :, 0]

dummy = np.zeros((NUM_SIMULATIONS * DAYS_FORWARD, 2))
dummy[:, 0] = simulated_scaled_returns.flatten()
real_log_returns = scaler.inverse_transform(dummy)[:, 0]
real_log_returns = real_log_returns.reshape(NUM_SIMULATIONS, DAYS_FORWARD)

equity_curves = np.zeros((NUM_SIMULATIONS, DAYS_FORWARD + 1))
equity_curves[:, 0] = STARTING_CAPITAL

for t in range(1, DAYS_FORWARD + 1):
    daily_multiplier = np.exp(real_log_returns[:, t-1])
    equity_curves[:, t] = equity_curves[:, t-1] * daily_multiplier

final_values = equity_curves[:, -1]
expected_value = np.mean(final_values)

period_return = (expected_value - STARTING_CAPITAL) / STARTING_CAPITAL
annualized_return = ((1 + period_return) ** (252 / DAYS_FORWARD)) - 1

print(f"\nWorst Case Scenario: ${np.min(final_values):,.2f}")
print(f"Average Expected Value: ${expected_value:,.2f}")
print(f"Best Case Scenario: ${np.max(final_values):,.2f}")
print(f"Probability of Profit: {(np.sum(final_values > STARTING_CAPITAL) / NUM_SIMULATIONS) * 100:.2f}%")
print(f"Expected Annual Yield: {annualized_return * 100:.2f}%")

plt.figure(figsize=(10, 6))
for i in range(100):
    plt.plot(equity_curves[i], color='blue', alpha=0.1)

plt.plot(np.mean(equity_curves, axis=0), color='red', linewidth=2, label="Expected Portfolio Value")

plt.title("Monte Carlo Simulation: $10,000 over next 20 Trading Days")
plt.xlabel("Days into the Future")
plt.ylabel("Portfolio Value ($)")
plt.axhline(STARTING_CAPITAL, color='black', linestyle='--', label="Breakeven ($10,000)")
plt.legend()
plt.grid(True)
plt.savefig("monte_carlo_10k.png", dpi=300, bbox_inches='tight')
print("Simulation graph saved successfully as 'monte_carlo_10k.png'")
