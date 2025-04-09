import numpy as np
from .black_scholes import simulate_gbm
from sklearn.linear_model import LinearRegression

def monte_carlo_american_option(S0, K, mu, sigma, T, dt, n_simulations, r, option_type="call"):
    #S0: Initial Stock Price
    #K: Strike Price
    #mu: Drift
    #sigma: Volatility
    #T: Time to maturity
    #dt: Time step (daily)
    #n_simulations: number of simulations
    #r: Risk free rate
    #option_type: The option type, "call" or "put"

    n_steps = int(T/dt)
    t, S = simulate_gbm(S0, mu, sigma, T, dt, n_simulations)
    payoff = np.zeros((n_simulations, n_steps))
    if option_type == "call":
        payoff[:, -1] = np.maximum(S[:, -1] - K, 0)  # At maturity for call
    elif option_type == "put":
        payoff[:, -1] = np.maximum(K - S[:, -1], 0)  # At maturity for put

    # Backward induction to decide whether to exercise at each step
    for i in range(n_steps - 2, -1, -1):
        # Discount factor
        discount = np.exp(-r * dt)

        # Regression to calculate continuation value
        in_the_money = (payoff[:, i + 1] > 0)  # Only consider paths where option is in-the-money
        X = S[in_the_money, i].reshape(-1, 1)  # Feature: Stock price
        y = payoff[in_the_money, i + 1] * discount  # Target: discounted payoff

        if len(X) > 1:  # If there are in-the-money paths
            model = LinearRegression()
            model.fit(X, y)  # Fit linear regression
            continuation_value = model.predict(X)  # Predict continuation value
        else:
            continuation_value = np.zeros_like(y)

        # If exercising is better, update the payoff
        if option_type == "call":
            payoff[in_the_money, i] = np.maximum(S[in_the_money, i] - K, continuation_value)
        elif option_type == "put":
            payoff[in_the_money, i] = np.maximum(K - S[in_the_money, i], continuation_value)

    # Calculate the option price as the discounted average of payoffs
    option_price = np.exp(-r * T) * np.mean(payoff[:, 0])  # Discounted average payoff at time 0
    return option_price, payoff[:, 0]  # Return all payoffs at time 0

