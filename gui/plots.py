# plots.py
import matplotlib.pyplot as plt
from models.black_scholes import simulate_gbm
from models.heston import simulate_heston
from models.sabr import simulate_sabr


def plot_paths(t, paths):
    fig, ax = plt.subplots()
    for path in paths:
        ax.plot(t, path, alpha=0.7)
    ax.set_title("Simulated Stock Price Paths (GBM)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True)
    return fig


def plot_heston_paths(S0, v0, mu, kappa, theta, xi, rho, T, dt, n_paths):
    t, s, _ = simulate_heston(S0, v0, mu, kappa, theta, xi, rho, T, dt, n_paths)
    fig, ax = plt.subplots()
    for i in range(min(5, len(s))):
        ax.plot(t, s[i])
    ax.set_title("Heston Model - Simulated Stock Prices")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True)
    return fig


def plot_sabr_paths(S0, alpha, beta, rho, nu, T, dt, n_paths):
    t, s, _ = simulate_sabr(S0, alpha, beta, rho, nu, T, dt, n_paths)
    fig, ax = plt.subplots()
    for i in range(min(5, len(s))):
        ax.plot(t, s[i])
    ax.set_title("SABR Model - Simulated Stock Prices")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True)
    return fig
