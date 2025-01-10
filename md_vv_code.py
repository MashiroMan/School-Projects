import numpy as np
import scipy as sp

Np = 200
T = .1
dt = .01
Nsteps = np.floor(100/dt).astype(int)
gamma = 1
kB = 1.38e-23
m = 1

time = np.linspace(0, Nsteps*dt, Nsteps)

Energy = np.zeros(Nsteps)
Radius = np.zeros(Nsteps)
QEnergy = np.zeros(Nsteps)
KinEnergy = np.zeros(Nsteps)

# Seed for random number generation
# Particle positions
X = np.random.rand(Np,3) - 0.5
V = np.random.normal(0,np.sqrt(T/m),[Np,3])
#V = np.zeros([Np,3])

#Kinetic energy computation
def Kinetic_Energy(V):
    return 0.5*np.sum(V*V)*m

#Classical energy computation (placeholder for the actual implementation)
def Class_Energy(X, Np, T):
    # Classical Energy
    r = np.sqrt(np.sum(X*X,axis = 1))
    #maybe add mass in here
    Eclass = -2*np.sum(sp.special.erf(np.sqrt(0.25*np.pi*np.pi*Np/T)*r)/(Np*r))
    return Eclass

#Quantum energy computation (placeholder for the actual implementation)
def Quantum_Energy(X, Np, T):
    # Quantum Energy
    r2 = np.sum((X-np.roll(X,1,axis=0))**2,axis = 1)
    Equant = (T**2)*m*Np/2*np.sum(r2)
    return Equant

#just spring terms
# def acceleration(X, Np, T):
#     a = - 2*m*Np*(T**2)*X+m*Np*(T**2)*(np.roll(X,1,axis=0)+np.roll(X,-1,axis=0))
#     return a/m

#acceleration computation
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

#Velocity Verlet with Langevin thermostat
# def velocity_verlet(X, V, Np, T, dt):
#     # Generate random force
#     R = np.random.normal(0, np.sqrt(2 * gamma * kB * T * dt), X.shape)

#     # Update velocity
#     v_half = V + (acceleration(X, Np, T)- gamma*V + R)*dt/2

#     # Update position
#     X_new = X + v_half*dt

#     # Generate new random force for the next half-step
#     R_new = np.random.normal(0, np.sqrt(2 * gamma * kB * T * dt), X.shape)

#     # Update velocity
#     V_new = v_half + (acceleration(X_new, Np, T)- gamma * v_half + R_new)*dt/2

#     return X_new, V_new

def velocity_verlet(X, V, Np, T, dt):

    # Update velocity
    v_half = V + (acceleration(X, Np, T))*dt/2

    # Update position
    X_new = X + v_half*dt

    # Update velocity
    V_new = v_half + (acceleration(X_new, Np, T))*dt/2

    return X_new, V_new


#main loop
for i in range(Nsteps):
    QEnergy[i] = Quantum_Energy(X, Np, T)
    Energy[i] = Class_Energy(X, Np, T) + Quantum_Energy(X, Np, T) + Kinetic_Energy(V)
    KinEnergy[i] = Kinetic_Energy(V)
    Radius[i] = np.sqrt(np.sum(X*X)/Np)
    X, V = velocity_verlet(X, V, Np, T, dt)

   # print(X[1,:])
    



#Plot results with time as the x-axis
energy_max = np.max(Energy)

import matplotlib.pyplot as plt
plt.plot(time ,Energy, scaley=False)
plt.xlabel('time')
plt.ylabel('Energy')
plt.ylim(0, 2*energy_max)
plt.grid()
plt.title('Energy vs time (MD)')

plt.plot(time, KinEnergy, scaley=False)
plt.show()


# plt.plot(Radius)
# plt.xlabel('MC steps')
# plt.ylabel('Radius')
# plt.grid()
# plt.title('Radius vs MC steps')
# plt.show()

# plt.plot(QEnergy)
# plt.xlabel('time')
# plt.ylabel('Quantum Energy')
# plt.grid()
# plt.title('Quantum Energy vs time (MD)')
# plt.show()

# print(Class_Energy(X, Np, T)+Quant_Energy(X, Np, T))


# X, V = velocity_verlet(X, V, Np, T, dt)

# print(Class_Energy(X, Np, T)+Quant_Energy(X, Np, T))

# print(Kinetic_Energy(V))
# print(1.5*Np*T)