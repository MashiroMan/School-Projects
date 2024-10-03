import numpy as np
import scipy as sp

Np = 200
#Nsteps = int(3e4)
Nsteps = 10000
T = 20000.0
dt = 0.001

Energy = np.zeros(Nsteps)
Radius = np.zeros(Nsteps)
QEnergy = np.zeros(Nsteps)

# Seed for random number generation
# Particle positions
X = np.random.rand(Np,3) - 0.5
V = np.random.normal(0,np.sqrt(3*T),[Np,3])/10000
#V = np.zeros([Np,3])

#Kinetic energy computation
def Kinetic_Energy(V):
    return 0.5*np.sum(V*V)

#Classical energy computation (placeholder for the actual implementation)
def Class_Energy(X, Np, T):
    # Classical Energy
    r = np.sqrt(np.sum(X*X,axis = 1))
    Eclass = -2*np.sum(sp.special.erf(np.sqrt(0.25*np.pi*np.pi*Np/T)*r)/(Np*r))
    return Eclass

#Quantum energy computation (placeholder for the actual implementation)
def Quantum_Energy(X, Np, T):
    # Quantum Energy
    r2 = np.sum((X-np.roll(X,1,axis=0))**2,axis = 1)
    Equant = Np*np.sum(r2)/(4.0*T*T)
    return Equant

#acceleration computation
def acceleration(X, Np, T):
    r = np.sqrt(np.sum(X*X, axis=1))
    #Classical Part 1
    a = -2*sp.special.erf(np.sqrt(0.25*np.pi*np.pi*Np/T)*r)/(Np*r*r*r)
    #Classical Part 2
    a = a + 2*np.sqrt(np.pi/Np/T)*np.exp(-0.25*np.pi*np.pi*Np/T*r*r)/(r*r)
    #Quantum Part 1
    a = a - Np/(T**2)
    #Re-shape a to to multiply with X
    a = a.reshape(Np,1)
    a = X*a+Np*(np.roll(X,1,axis=0)+np.roll(X,-1,axis=0))/2/(T*T)
    return a

#Velocity Verlet
def velocity_verlet(X, V, Np, T, dt):
    v_half = V + acceleration(X, Np, T)*dt/2
    X_new = X + v_half*dt
    V_new = v_half + acceleration(X_new, Np, T)*dt/2

    return X_new, V_new


#main loop
for i in range(Nsteps):
    QEnergy[i] = Quantum_Energy(X, Np, T)
    Energy[i] = Class_Energy(X, Np, T) + Quantum_Energy(X, Np, T)
    Radius[i] = np.sqrt(np.sum(X*X)/Np)
    X, V = velocity_verlet(X, V, Np, T, dt)

   # print(X[1,:])
    



#Plot results
import matplotlib.pyplot as plt
plt.plot(Energy)
plt.xlabel('MC steps')
plt.ylabel('Energy')
plt.grid()
plt.title('Energy vs MC steps')
plt.show()


# plt.plot(Radius)
# plt.xlabel('MC steps')
# plt.ylabel('Radius')
# plt.grid()
# plt.title('Radius vs MC steps')
# plt.show()

# plt.plot(QEnergy)
# plt.xlabel('MC steps')
# plt.ylabel('Quantum Energy')
# plt.grid()
# plt.title('Quantum Energy vs MC steps')
# plt.show()

# print(Class_Energy(X, Np, T)+Quant_Energy(X, Np, T))


# X, V = velocity_verlet(X, V, Np, T, dt)

# print(Class_Energy(X, Np, T)+Quant_Energy(X, Np, T))

# print(Kinetic_Energy(V))
# print(1.5*Np*T)