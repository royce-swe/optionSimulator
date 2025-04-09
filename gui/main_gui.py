import tkinter as tk
from models.black_scholes import simulate_gbm
from gui.plots import plot_paths, plot_heston_paths, plot_sabr_paths
import yfinance as yf
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Fetch historical AAPL data
df = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
df = df.xs("AAPL", level=1, axis=1)
df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
mu_val = df['LogReturns'].mean() * 252
sigma_val = df['LogReturns'].std() * np.sqrt(252)

# GUI setup
root = tk.Tk()
root.title("Stochastic Model Simulator")
root.geometry("1200x1000")  # Wider instead of taller

# Create left and right frames
left_frame = tk.Frame(root, padx=10)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = tk.Frame(root, padx=10)
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

canvas = None

def labeled_entry(parent, label, default=""):
    tk.Label(parent, text=label).pack()
    entry = tk.Entry(parent)
    entry.insert(0, default)
    entry.pack()
    return entry

# GBM Inputs
entry_S0 = labeled_entry(left_frame, "Initial Price (S0)", "100")
entry_mu = labeled_entry(left_frame, "Drift (μ)", f"{mu_val:.4f}")
entry_sigma = labeled_entry(left_frame, "Volatility (σ)", f"{sigma_val:.4f}")
entry_T = labeled_entry(left_frame, "Time Horizon (T)", "2")
entry_dt = labeled_entry(left_frame, "Time Step (dt)", "0.01")
entry_paths = labeled_entry(left_frame, "Number of Paths", "50")

tk.Label(left_frame, text="--- Heston Model Parameters ---").pack()
entry_v0 = labeled_entry(left_frame, "Initial Variance (v0)", "0.04")
entry_kappa = labeled_entry(left_frame, "Mean Reversion Rate (κ)", "2.0")
entry_theta = labeled_entry(left_frame, "Long-term Variance (θ)", "0.04")
entry_xi = labeled_entry(left_frame, "Volatility of Volatility (ξ)", "0.3")
entry_rho_heston = labeled_entry(left_frame, "Correlation (ρ)", "-0.7")

tk.Label(left_frame, text="--- SABR Model Parameters ---").pack()
entry_alpha = labeled_entry(left_frame, "Alpha", "0.04")
entry_beta = labeled_entry(left_frame, "Beta", "0.5")
entry_rho_sabr = labeled_entry(left_frame, "Correlation (ρ)", "0.0")
entry_nu = labeled_entry(left_frame, "Nu", "0.2")

def display_plot(fig):
    global canvas
    if canvas:
        canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def run_simulation():
    S0 = float(entry_S0.get())
    mu = float(entry_mu.get())
    sigma = float(entry_sigma.get())
    T = float(entry_T.get())
    dt = float(entry_dt.get())
    n_paths = int(entry_paths.get())
    t, paths = simulate_gbm(S0, mu, sigma, T, dt, n_paths)
    fig = plot_paths(t, paths)
    display_plot(fig)

def run_heston_simulation():
    fig = plot_heston_paths(
        S0=float(entry_S0.get()),
        v0=float(entry_v0.get()),
        mu=float(entry_mu.get()),
        kappa=float(entry_kappa.get()),
        theta=float(entry_theta.get()),
        xi=float(entry_xi.get()),
        rho=float(entry_rho_heston.get()),
        T=float(entry_T.get()),
        dt=float(entry_dt.get()),
        n_paths=int(entry_paths.get())
    )
    display_plot(fig)

def run_sabr_simulation():
    fig = plot_sabr_paths(
        S0=float(entry_S0.get()),
        alpha=float(entry_alpha.get()),
        beta=float(entry_beta.get()),
        rho=float(entry_rho_sabr.get()),
        nu=float(entry_nu.get()),
        T=float(entry_T.get()),
        dt=float(entry_dt.get()),
        n_paths=int(entry_paths.get())
    )
    display_plot(fig)

tk.Button(left_frame, text="Run GBM Simulation", command=run_simulation).pack(pady=5)
tk.Button(left_frame, text="Run Heston Simulation", command=run_heston_simulation).pack(pady=5)
tk.Button(left_frame, text="Run SABR Simulation", command=run_sabr_simulation).pack(pady=5)

root.mainloop()
