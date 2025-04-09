#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

// Function to simulate a Geometric Brownian Motion
vector<double> simulateGBM(double S0, double mu, double sigma, double T, double dt, int N) {
    // Number of time steps
    int num_steps = int(T / dt);

    // Vector to store the simulated path
    vector<double> path(num_steps);

    // Random number generator and normal distribution for Brownian motion
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist(0, 1); // Standard normal distribution

    // Initial stock price
    path[0] = S0;

    // Simulate the GBM path
    for (int i = 1; i < num_steps; ++i) {
        // GBM formula: S(t+1) = S(t) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
        double dW = dist(gen) * sqrt(dt); // Brownian increment
        path[i] = path[i-1] * exp((mu - 0.5 * sigma * sigma) * dt + sigma * dW);
    }

    return path;
}

int main() {
    std::cout << "Program started!" << std::endl;
    // Parameters for the GBM simulation
    double S0 = 100;   // Initial stock price
    double mu = 0.05;  // Drift (expected return)
    double sigma = 0.2; // Volatility
    double T = 1.0;    // Time horizon (1 year)
    double dt = 0.01;  // Time step (1 day)
    int N = 1000;      // Number of simulations (paths)

    // Simulate the GBM path
    vector<double> path = simulateGBM(S0, mu, sigma, T, dt, N);

    // Print the simulated path (stock prices at each time step)
    for (size_t i = 0; i < path.size(); ++i) {
        cout << "Time " << i * dt << " : " << path[i] << endl;
    }

    return 0;
}
