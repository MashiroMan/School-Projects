import numpy as np
import langevin
import matplotlib.pyplot as plt

# Parameters
gamma_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
temperature = .1
num_particles = 200
mass = 1/num_particles
trasient_steps = 10000
num_steps = 10000
dt = 0.01

# Function to calculate variance of temperature
def gather_stats(temperature_data):
    return np.var(temperature_data), np.mean(temperature_data)

# Function to initialize the system
def initialize_system(num_particles, temperature):
    positions = np.random.rand(num_particles, 3) - 0.5
    velocities = np.random.normal(0, np.sqrt(temperature/mass), [num_particles, 3])
    return positions, velocities

# Function to run the simulation
def run_simulation(positions, velocities, gamma):
    for i in range(trasient_steps):
        positions, velocities = langevin.euler_maruyama(positions, velocities, num_particles, mass, temperature, dt, gamma)
    temperature_data = []
    for i in range(num_steps):
        positions, velocities = langevin.euler_maruyama(positions, velocities, num_particles, mass, temperature, dt, gamma)
        temperature_data.append(langevin.Kinetic_Energy(velocities, mass) / (1.5 * num_particles))
    return temperature_data

# Run simulations and collect data
variances = []
means = []
for gamma in gamma_values:
    positions, velocities = initialize_system(num_particles, temperature)
    temperature_data = run_simulation(positions, velocities, gamma)
    var, mean = gather_stats(temperature_data)
    variances.append(var)
    means.append(mean)

epsilon = []

for i in range(len(gamma_values)):
    eps = temperature - means[i]
    if eps < 0:
        eps = -eps + variances[i]
    else:
        eps = eps + variances[i]
    epsilon.append(eps/temperature)

# Plot results
# var_max = np.max(variances)

# plt.figure()
# plt.plot(gamma_values, variances, marker='o', scaley=False)
# plt.xlabel('Gamma')
# plt.ylabel('Variance of Temperature')
# plt.ylim(0, 2*var_max)
# plt.title('Variance of Temperature vs Gamma')
# plt.grid(True)
# plt.show()

# mean_max = np.max(means)

# plt.figure()
# plt.plot(gamma_values, means, marker='o', scaley=False)
# plt.xlabel('Gamma')
# plt.ylabel('Mean of Temperature')
# plt.ylim(0, 2*mean_max)
# plt.title('Mean of Temperature vs Gamma')
# plt.grid(True)
# plt.show()


epsilon_max = np.max(epsilon)
plt.figure()
plt.plot(gamma_values, epsilon, marker='o')
plt.xlabel('Gamma')
plt.ylabel('Epsilon')
plt.yscale('log')
plt.title('Epsilon vs Gamma')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(gamma_values, epsilon, marker='o', scaley=False)
plt.xlabel('Gamma')
plt.ylabel('Epsilon')
plt.ylim(0, 2*epsilon_max)
plt.title('Epsilon vs Gamma')
plt.grid(True)
plt.show()