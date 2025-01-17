import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Langevin:
    def __init__(self, params):
        """
        Initialize the Langevin simulation with given parameters.

        params: dict containing simulation parameters.
        """
        self.N = params['N']
        self.T = params['T']
        self.dt = params['dt']
        self.Nsteps = np.floor(params['time']/self.dt).astype(int)
        self.gamma = params['gamma']
        self.m = params['m']
        self.miu = self.m/self.N
        self.kB = params.get('kB', 1.38e-23)  # Default to Boltzmann constant

        # Initialize arrays
        self.time = np.linspace(0, self.Nsteps * self.dt, self.Nsteps)
        self.Energy = np.zeros(self.Nsteps)
        self.Radius = np.zeros(self.Nsteps)
        self.QEnergy = np.zeros(self.Nsteps)
        self.KinEnergy = np.zeros(self.Nsteps)

        # Initialize particle positions and velocities
        self.X = np.random.rand(self.Np, 3) - 0.5
        self.V = np.random.normal(0, np.sqrt(self.T / self.m), [self.Np, 3])

    def kinetic_energy(self):
        return 0.5 * np.sum(self.V * self.V) * self.m

    def classical_energy(self):
        r = np.sqrt(np.sum(self.X * self.X, axis=1))
        Eclass = -2 * np.sum(sp.special.erf(np.sqrt(0.25 * np.pi**2 * self.Np / self.T) * r) / (self.Np * r))
        return Eclass

    def quantum_energy(self):
        r2 = np.sum((self.X - np.roll(self.X, 1, axis=0))**2, axis=1)
        Equant = (self.T**2) * self.m * self.Np / 2 * np.sum(r2)
        return Equant

    def acceleration(self):
        r = np.sqrt(np.sum(self.X * self.X, axis=1))
        a = -sp.special.erf(np.sqrt(0.25 * np.pi**2 * self.Np / self.T) * r)
        a += r * np.sqrt(np.pi * self.Np / self.T) * np.exp(-0.25 * np.pi**2 * self.Np / self.T * r**2)
        a = 2 / self.Np * a / (r**3) - 2 * self.m * self.Np * (self.T**2)
        a = a.reshape(self.Np, 1) * self.X + self.m * self.Np * (self.T**2) * (
            np.roll(self.X, 1, axis=0) + np.roll(self.X, -1, axis=0)
        )
        return a / self.m

    def euler_maruyama(self):
        a = self.acceleration()
        self.X = self.X + self.V*self.dt

    def velocity_verlet(self):
        R = np.random.normal(0, np.sqrt(2 * self.gamma * self.kB * self.T * self.dt), self.X.shape)
        v_half = self.V + (self.acceleration() - self.gamma * self.V + R) * self.dt / 2
        self.X += v_half * self.dt
        R_new = np.random.normal(0, np.sqrt(2 * self.gamma * self.kB * self.T * self.dt), self.X.shape)
        self.V = v_half + (self.acceleration() - self.gamma * v_half + R_new) * self.dt / 2

    def run_simulation(self):
        """
        Run the Langevin simulation.
        """
        for i in range(self.Nsteps):
            self.QEnergy[i] = self.quantum_energy()
            self.Energy[i] = (
                self.classical_energy() + self.quantum_energy() + self.kinetic_energy()
            )
            self.KinEnergy[i] = self.kinetic_energy()
            self.Radius[i] = np.sqrt(np.sum(self.X * self.X) / self.Np)
            self.velocity_verlet()

    def plot_results(self):
        """
        Plot energy and radius over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.Energy, label="Total Energy", color="blue")
        plt.plot(self.time, self.KinEnergy, label="Kinetic Energy", color="green")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title("Energy vs Time")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.Radius, label="Radius", color="purple")
        plt.xlabel("Time")
        plt.ylabel("Radius")
        plt.title("Radius vs Time")
        plt.grid()
        plt.show()

# Example usage
params = {
    'Np': 200,
    'T': 0.1,
    'dt': 0.01,
    'time': 100,  # Total simulation time
    'gamma': 1,
    'm': 1
}

langevin_sim = Langevin(params)
langevin_sim.run_simulation()
langevin_sim.plot_results()
