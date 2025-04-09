import numpy as np


def simulate_heston(S0, v0, mu, kappa, theta, xi, rho, T, dt, n_paths):
    N = int(T / dt)
    t = np.linspace(0, T, N)

    S = np.zeros((n_paths, N))
    v = np.zeros((n_paths, N))

    S[:, 0] = S0
    v[:, 0] = v0

    for i in range(1, N):
        z1 = np.random.standard_normal(n_paths)
        z2 = np.random.standard_normal(n_paths)

        # Correlate z2 with z1 using Cholesky method
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2

        # Make sure variance stays non-negative
        v[:, i] = np.maximum(v[:, i - 1] + kappa * (theta - v[:, i - 1]) * dt + xi * np.sqrt(v[:, i - 1] * dt) * z2, 0)
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * v[:, i - 1]) * dt + np.sqrt(v[:, i - 1] * dt) * z1)

    return t, S, v
