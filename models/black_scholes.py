import numpy as np
from scipy.stats import norm

def simulate_gbm(S0, mu, sigma, T, dt, n_paths): #S0: initial stock price, mu: drift, sigma: volatility, T: total time horizon, dt: Time step, n_paths: Number of simulation paths
     N = int(T/ dt) #Calculates how many time steps to simulate
     t = np.linspace(0, T, N) #Generates a time grid from t = 0 to t = T with N points
     paths = np.zeros((n_paths, N)) #Creates a matix to store simulated prices, n_paths rows: how many diferent stock price paths we simulate, N columns: the number of time steps
     paths[:, 0] = S0 #set the initial stock price at time ) for all paths
     for i in range(1, N): #Generate n_paths random numbers z ~ N(0,1), Plug into the GBM formula to calculate the price at the next time step
         z = np.random.standard_normal(n_paths)
         paths[:, i] = paths[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
     return t, paths

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * sp.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)