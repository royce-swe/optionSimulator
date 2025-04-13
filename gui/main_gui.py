import sys
import os

# Adds the root "optionSimulation" folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
from models.black_scholes import simulate_gbm
from models.monte_carlo import monte_carlo_american_option
from gui.plots import plot_paths, plot_heston_paths, plot_sabr_paths, plot_black_scholes_3d, plot_monte_carlo_distribution
import yfinance as yf
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Fetch historical AAPL data
df = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
df = df.xs("AAPL", level=1, axis=1)
df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
mu_val = df['LogReturns'].mean() * 252 #Calculate Drift
sigma_val = df['LogReturns'].std() * np.sqrt(252) #Calculate Volitility

# GUI setup
root = tk.Tk()
root.title("Stochastic Model Simulator")
root.geometry("1200x1200")

# Create a canvas and scrollbar
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the canvas to hold all the content
scrollable_frame = tk.Frame(canvas)

# Add the scrollable frame to the canvas
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Pack the canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Configure the scroll region when the frame size changes
def configure_scroll_region(event):
    scrollable_frame.configure(scrollregion=canvas.bbox("all"))

scrollable_frame.bind("<Configure>", configure_scroll_region)

# Create left and right frames inside the scrollable frame
left_frame = tk.Frame(scrollable_frame, padx=10)
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
tk.Label(left_frame, text="--- Geometric Brownian Motion ---").pack()
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

tk.Label(left_frame, text="--- Black-Scholes Model ---").pack()
entry_S0 = labeled_entry(left_frame, "Initial Price (S0)", "100")
entry_T_min = labeled_entry(left_frame, "Minimum Time to Maturity (T_min)", "0.1")
entry_T_max = labeled_entry(left_frame, "Maximum Time to Maturity (T_max)", "2.0")
entry_r = labeled_entry(left_frame, "Risk-Free Rate (r)", "0.05")
entry_sigma = labeled_entry(left_frame, "Volatility (σ)", "0.2")
option_type_var = tk.StringVar(value="call")  # Default is "call"
option_menu = tk.OptionMenu(left_frame, option_type_var, "call", "put")
option_menu.pack(pady=5)

tk.Label(left_frame, text="--- American Option Monte Carlo ---").pack()
entry_K = labeled_entry(left_frame, "Strike Price (K)", "100")
entry_T_american = labeled_entry(left_frame, "Time to Maturity (T)", "1.0")
entry_r_american = labeled_entry(left_frame, "Risk-Free Rate (r)", "0.80")
entry_n_sim = labeled_entry(left_frame, "Number of Simulations", "100000")
american_option_type_var = tk.StringVar(value="call")
option_menu_am = tk.OptionMenu(left_frame, american_option_type_var, "call", "put")
option_menu_am.pack(pady=5)


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

def run_black_scholes_simulation():
    S0 = float(entry_S0.get())  # Get initial stock price
    r = float(entry_r.get())  # Get risk-free rate
    sigma = float(entry_sigma.get())  # Get volatility
    option_type = option_type_var.get()
    
    # Get the minimum and maximum time to maturity from the GUI
    T_min = float(entry_T_min.get())  # Minimum time to maturity
    T_max = float(entry_T_max.get())  # Maximum time to maturity

    # Call the 3D plotting function with dynamic T_min and T_max
    fig = plot_black_scholes_3d(S=S0, r=r, sigma=sigma, T_min=T_min, T_max=T_max, option_type=option_type)
    display_plot(fig)

def run_american_option_simulation():
    S0 = float(entry_S0.get())
    K = float(entry_K.get())
    T = float(entry_T_american.get())
    r = float(entry_r_american.get())
    n_sim = int(entry_n_sim.get())
    sigma = float(entry_sigma.get())
    dt = float(entry_dt.get())
    option_type = american_option_type_var.get()

    mu = r  # << IMPORTANT: risk-neutral drift for pricing

    price, payoffs = monte_carlo_american_option(
        S0=S0, K=K, mu=mu, sigma=sigma, T=T, dt=dt,
        n_simulations=n_sim, r=r, option_type=option_type
    )

    fig = plot_monte_carlo_distribution(payoffs, option_type=option_type)
    display_plot(fig)



# Buttons for simulations
tk.Button(left_frame, text="Run GBM Simulation", command=run_simulation).pack(pady=5)
tk.Button(left_frame, text="Run Heston Simulation", command=run_heston_simulation).pack(pady=5)
tk.Button(left_frame, text="Run SABR Simulation", command=run_sabr_simulation).pack(pady=5)
tk.Button(left_frame, text="Run Black-Scholes Simulation", command=run_black_scholes_simulation).pack(pady=5)
tk.Button(left_frame, text="Run Monte Carlo Simulation", command=run_american_option_simulation).pack(pady=5)

def on_closing():
    root.quit()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
