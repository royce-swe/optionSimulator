import numpy as np
from .black_scholes import simulate_gbm
from sklearn.linear_model import LinearRegression

def monte_carlo_american_option(S0, K, mu, sigma, T, dt, n_simulations, r, option_type="call"):
    n_steps = int(T / dt)
    t, S = simulate_gbm(S0, mu, sigma, T, dt, n_simulations)
    payoff = np.zeros((n_simulations, n_steps))

    # Initialize terminal payoffs
    if option_type == "call":
        payoff[:, -1] = np.maximum(S[:, -1] - K, 0)
    elif option_type == "put":
        payoff[:, -1] = np.maximum(K - S[:, -1], 0)

    # Backward induction
    for i in range(n_steps - 2, -1, -1):
        discount = np.exp(-r * dt)

        if option_type == "call":
            in_the_money = S[:, i] > K
            intrinsic_value = S[in_the_money, i] - K
        elif option_type == "put":
            in_the_money = S[:, i] < K
            intrinsic_value = K - S[in_the_money, i]

        # Regression only for in-the-money paths
        X = S[in_the_money, i].reshape(-1, 1)
        y = payoff[in_the_money, i + 1] * discount

        if len(X) > 1:
            model = LinearRegression()
            model.fit(X, y)
            continuation_value = model.predict(X)
        else:
            continuation_value = np.zeros_like(y)

        # Exercise decision
        exercise = intrinsic_value > continuation_value
        # Update payoffs for in-the-money paths
        payoff[in_the_money, i] = np.where(
            exercise,
            intrinsic_value,
            payoff[in_the_money, i + 1] * discount
        )

        # Out-of-the-money paths: carry forward discounted value
        not_in_the_money = ~in_the_money
        payoff[not_in_the_money, i] = payoff[not_in_the_money, i + 1] * discount

    option_price = np.exp(-r * T) * np.mean(payoff[:, 0])
    return option_price, payoff[:, 0]
