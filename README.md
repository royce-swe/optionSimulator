# 🧠 Option Pricing Simulator

A powerful and interactive GUI-based tool to visualize and simulate option prices using cutting-edge stochastic models. Built with Python and Tkinter, this project offers financial insight for students, researchers, and enthusiasts.

## 📈 Features

- **Black-Scholes Model** – Classic closed-form pricing for European call and put options.
- **Geometric Brownian Motion (GBM)** – Simulates asset price paths using standard stochastic processes.
- **Heston Model** – Captures stochastic volatility with mean-reverting variance.
- **SABR Model** – Incorporates stochastic volatility with asset price correlation; used for complex derivatives pricing.
- **American Option Monte Carlo Simulation** – Uses Longstaff-Schwartz regression to simulate early exercise behavior and optimal stopping.

## 💻 Technologies Used

- Python 3
- Tkinter (GUI)
- Matplotlib (Plotting)
- NumPy, SciPy
- scikit-learn (for regression in Monte Carlo)

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/royce-swe/optionSimulator.git
   cd optionSimulator
   python gui/main_gui.py
