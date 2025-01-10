import numpy as np
import scipy as sp


Np = 200
T = .1
dt = .01
avg_size = np.floor(10/dt).astype(int)
gamma = .2
m = 1
epsilon = .01



X = np.random.rand(Np,3) - 0.5
V = np.random.normal(0,np.sqrt(T/m),[Np,3])

def Kinetic_Energy(V, m, Np):
    return 0.5*np.sum(V*V)*m/Np

def acceleration(X, Np, m, T):
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
    return a/m*Np

def euler_maruyama(X, V, Np, m, T, dt, gamma):
    a = acceleration(X, Np, m, T)
    X = X + V*dt
    V = V + (a-gamma*V)*dt + np.random.normal(0,np.sqrt(2*gamma*T/m*Np*dt),[Np,3])
    return X, V

# main loop

#transient stage
for i in range(10000):
    X, V = euler_maruyama(X, V, Np, m, T, dt, gamma)

# thermal equilibrium

Nsteps = np.floor(100/dt).astype(int)
temp = np.zeros(Nsteps)


for i in range(Nsteps):
    X, V = euler_maruyama(X, V, Np, m, T, dt, gamma)
    temp[i] = Kinetic_Energy(V, m, Np)/(1.5*Np)



import matplotlib.pyplot as plt

time = np.linspace(0, Nsteps*dt, Nsteps)

temp_max = np.max(temp)
plt.plot(time, temp, scaley=False)
plt.xlabel('time after transient stage')
plt.ylabel('Temperature')
plt.ylim(0, 2*temp_max)
plt.grid()
plt.title('Temp vs time after transient stage')
plt.axhline(y=T, color='r', linestyle='--')
plt.show()



