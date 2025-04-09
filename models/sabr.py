import numpy as np

"""
SABR Model (Stochastic Alpha Beta Rho):
A stochastic volatility model commonly used in interest rate and FX derivatives markets.
Characterized by a flexible power-law (beta) term, it models both the forward price and its volatility as correlated stochastic processes, enabling it to capture skew and smile effects in implied volatility surfaces.
"""


def simulate_sabr(S0, alpha, beta, rho, nu, T, dt, n_paths):
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    s = np.zeros((n_paths, n_steps))  # Stock price paths
    v = np.zeros((n_paths, n_steps))  # Volatility paths
    # Initial conditions
    s[:, 0] = S0
    v[:, 0] = alpha  # Initial volatility (alpha can be thought of as initial volatility)
    for i in range(1, n_steps):
        # Generate Brownian motions for price and volatility
        dW1 = np.random.normal(0, np.sqrt(dt), size=n_paths)  # Brownian motion for stock price
        dW2 = rho * dW1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), size=n_paths)  # Brownian motion for volatility
        # Update the volatility using the SABR model with beta
        v[:, i] = np.maximum(v[:, i - 1] + nu * v[:, i - 1] ** beta * dW2, 0.0)  # Volatility path: includes beta also Maximum prevents negative volatility
        # Update the stock price using the volatility path
        s[:, i] = s[:, i - 1] * np.exp(alpha * np.sqrt(dt) * (v[:, i] + dW1))  # Stock price path
    return t, s, v  # Return both stock price and volatility paths
