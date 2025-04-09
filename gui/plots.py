# plots.py
import matplotlib.pyplot as plt
from models.black_scholes import simulate_gbm
from models.heston import simulate_heston
from models.sabr import simulate_sabr
from models.black_scholes import black_scholes_price
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 


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

def plot_black_scholes_3d(S, r, sigma, T_min, T_max, option_type="call"):
    K_vals = np.linspace(50, 150, 100)  # Strike prices
    T_vals = np.linspace(T_min, T_max, 100)  # Time to maturity (from user input)

    # Create a meshgrid for K and T
    K_grid, T_grid = np.meshgrid(K_vals, T_vals)

    # Calculate the option prices for each (K, T) pair
    prices = np.array([[black_scholes_price(S, K, T, r, sigma, option_type) for K in K_vals] for T in T_vals])

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(K_grid, T_grid, prices, cmap='viridis')

    # Set labels and title
    ax.set_title(f"Black-Scholes Option Price vs Strike and Time to Maturity ({option_type.title()})")
    ax.set_xlabel("Strike Price (K)")
    ax.set_ylabel("Time to Maturity (T)")
    ax.set_zlabel("Option Price")
    ax.grid(True)
    return fig

"""
def plot_black_scholes_vs_strike(S, T, r, sigma, option_type="call"):
    K_vals = np.linspace(50, 150, 100)
    prices = [black_scholes_price(S, K, T, r, sigma, option_type) for K in K_vals]
    fig, ax = plt.subplots()
    ax.plot(K_vals, prices)
    ax.set_title(f"Black-Scholes Option Price vs Strike ({option_type.title()})")
    ax.set_xlabel("Strike Price (K)")
    ax.set_ylabel("Option Price")
    ax.grid(True)
    return fig
"""