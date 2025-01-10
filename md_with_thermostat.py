import numpy as np
import scipy as sp
import sys
import matplotlib.pyplot as plt

# Parameters
Np = 200
m = 1
T = 0.1

# Seed for random number generation

# Particle positions and velocity
X = np.random.rand(Np,3) - 0.5
V = np.random.normal(0,np.sqrt(T/m),[Np,3])

# Energy computation

def Kinetic_Energy(V):
    return 0.5*np.sum(V*V)*m/Np

def IE_Energy(X, Np, T):
    # Classical Energy
    r = np.sqrt(np.sum(X*X,axis = 1))
    #maybe add mass in here
    ie = -2*np.sum(sp.special.erf(np.sqrt(0.25*np.pi*np.pi*Np/T)*r)/(Np*r))
    return ie

def Spring_Energy(X, Np, T):
    # Quantum Energy
    r2 = np.sum((X-np.roll(X,1,axis=0))**2,axis = 1)
    spring = (T**2)*m*Np/2*np.sum(r2)
    return spring

# Acceleration due to potential

def acceleration(X, Np, T):
    r = np.sqrt(np.sum(X*X, axis=1))
    # Classical Part 1
    a = -sp.special.erf(np.sqrt(0.25*np.pi*np.pi*Np/T)*r)
    # Classical Part 2
    a = a + r*np.sqrt(np.pi*Np/T)*np.exp(-0.25*np.pi*np.pi*Np/T*r*r)
    # Quantum Part 1
    a = 2/Np*a/(r**3) - 2*m*Np*(T**2)
    # Re-shape a to to multiply with X
    a = a.reshape(Np,1)
    a = a*X+m*Np*(T**2)*(np.roll(X,1,axis=0)+np.roll(X,-1,axis=0))
    return a/m

# Thermalization

# Thermalization parameters (for now no epsilon, just simple transient stage)
dt = 0.01
gamma = 2
thermal_steps = 10000

def Euler_Maruyama(X, V, Np, T, dt):
    a = acceleration(X, Np, T)
    X = X + V*dt
    V = V + (a-gamma*V)*dt + np.random.normal(0,np.sqrt(2*gamma*T/m*dt),[Np,3])
    return X, V

# Thermalization loop

for i in range(thermal_steps):
    X, V = Euler_Maruyama(X, V, Np, T, dt)

print('Thermalization done, current temperature is', Kinetic_Energy(V)/(1.5*Np))

# Post_thermalization

# Parameters
dt = 0.01
Nsteps = np.floor(100/dt).astype(int)

# Energy arrays
Energy = np.zeros(Nsteps)
Radius = np.zeros(Nsteps)
QEnergy = np.zeros(Nsteps)
KinEnergy = np.zeros(Nsteps)

# Velocity Verlet

def Velocity_Verlet(X, V, Np, T, dt):
    a = acceleration(X, Np, T)
    X = X + V*dt + 0.5*a*dt*dt
    a_new = acceleration(X, Np, T)
    V = V + 0.5*(a+a_new)*dt
    return X, V

# Main loop

for i in range(Nsteps):
    QEnergy[i] = Spring_Energy(X, Np, T)
    Energy[i] = IE_Energy(X, Np, T) + Spring_Energy(X, Np, T) + Kinetic_Energy(V)
    KinEnergy[i] = Kinetic_Energy(V)
    Radius[i] = np.sqrt(np.sum(X*X)/Np)
    X, V = Velocity_Verlet(X, V, Np, T, dt)

# Plotting

time = np.linspace(0, Nsteps*dt, Nsteps)
energy_max = np.max(Energy)

# plt.plot(time, Energy, scaley=False)
# plt.xlabel('time')
# plt.ylabel('Energy')
# plt.ylim(0, 2*energy_max)
# plt.grid()
# plt.title('Energy vs time (MD)')
# plt.show()



temp = KinEnergy/(1.5*Np)
temp_max = np.max(temp)

# plt.plot(time, temp, scaley=False)
# plt.xlabel('time')
# plt.ylabel('Temperature')
# plt.ylim(0, 2*temp_max)
# plt.grid()
# plt.title('Temperature vs time (MD)')
# plt.show()

spring_energy_max = np.max(QEnergy)
ie_energy_max = np.max(IE_Energy(X, Np, T))

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs = axs.flatten()
# Energy plot
axs[0].plot(time, Energy)
axs[0].set_xlabel('time')
axs[0].set_ylabel('Energy')
axs[0].set_ylim(0, 2 * energy_max)
axs[0].grid()
axs[0].set_title('Energy vs time (MD)')

# Temperature plot
axs[1].plot(time, temp)
axs[1].set_xlabel('time')
axs[1].set_ylabel('Temperature')
axs[1].set_ylim(0, 2 * temp_max)
axs[1].grid()
axs[1].set_title('Temperature vs time (MD)')

# Spring Energy plot
axs[2].plot(time, QEnergy)
axs[2].set_xlabel('time')
axs[2].set_ylabel('Spring Energy')
axs[2].set_ylim(0, 2 * spring_energy_max)
axs[2].grid()
axs[2].set_title('Spring Energy vs time (MD)')

# IE Energy plot
axs[3].plot(time, [IE_Energy(X, Np, T)] * len(time))  # Assuming IE_Energy is constant over time
axs[3].set_xlabel('time')
axs[3].set_ylabel('IE Energy')
axs[3].set_ylim(0, 2 * ie_energy_max)
axs[3].grid()
axs[3].set_title('IE Energy vs time (MD)')

plt.tight_layout()
plt.show()


