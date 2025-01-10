import numpy as np
import scipy as sp

rejects = 0
# Parameters
Np = 200
#Nsteps = int(3e4)
Nsteps = 30000
T = 0.1
m = 1

Energy = np.zeros(Nsteps)
Radius = np.zeros(Nsteps)
QEnergy = np.zeros(Nsteps)


# Seed for random number generation
# Particle positions
X = np.random.rand(Np,3) - 0.5
#print(np.sum(X*X, axis=1))




# Classical energy computation (placeholder for the actual implementation)
def Class_Energy(X, Np, T):
    # Classical Energy
    r = np.sqrt(np.sum(X*X,axis = 1))
    #print(r.shape)
    Eclass = -2*np.sum(sp.special.erf(np.sqrt(0.25*np.pi*np.pi*Np/T)*r)/(Np*r))
    return Eclass


# Quantum energy computation (placeholder for the actual implementation)
def Quant_Energy(X, Np, T):
    # Quantum Energy
    r2 = np.sum((X-np.roll(X,1,axis=0))**2,axis = 1)
    Equant = (T**2)*m*Np/2*np.sum(r2)
    return Equant


# Metropolis Monte Carlo move (placeholder for the actual implementation)
def MCmove(X, Np, T):
    #compute old energy
    E_Old = T*(Class_Energy(X, Np,T) + Quant_Energy(X, Np,T))

    #choose a particle to move
    i = np.random.randint(Np)

    #store old position
    x_old = X[i, 0]
    y_old = X[i, 1]
    z_old = X[i, 2]
    #print(X_Old)

    #perturb the position
    X[i,0] += (0.2*np.random.rand() - 0.1)
    X[i,1] += (0.2*np.random.rand() - 0.1)
    X[i,2] += (0.2*np.random.rand() - 0.1)

    #compute new energy
    E_New = T*(Class_Energy(X, Np,T) + Quant_Energy(X, Np,T))

    #metroplis test
    if (E_New > E_Old):
        if (np.random.rand() < np.exp(-(E_New - E_Old)/T)):
        #reject
            X[i,0] = x_old
            X[i,1] = y_old
            X[i,2] = z_old
            global rejects
            rejects += 1
    
    return X


# Main program
for i in range(Nsteps):
    # print('Step: ', i, ', 1.5*Np/T: ', 1.5*Np/T , ', Classical Energy: ', Class_Energy(X, Np, T), ', Quantum Energy: ', Quant_Energy(X, Np, T))
    Energy[i] = Class_Energy(X, Np, T) + Quant_Energy(X, Np, T) + 1.5*Np*T
    Radius[i] = np.sqrt(np.sum(X*X)/Np)
    QEnergy[i] = Quant_Energy(X, Np, T)
    X = MCmove(X, Np, T)

# Plor results
import matplotlib.pyplot as plt
plt.plot(Energy)
plt.xlabel('MC steps')
plt.ylabel('Energy')
plt.grid()
plt.title('Energy vs MC steps')
plt.show()

# plt.plot(QEnergy)
# plt.xlabel('MC steps')
# plt.ylabel('Quantum Energy')
# plt.grid()
# plt.title('Quantum Energy vs MC steps')
# plt.show()


# plt.plot(Radius)
# plt.xlabel('MC steps')
# plt.ylabel('Radius')
# plt.grid()
# plt.title('Radius vs MC steps')
# plt.show()

# print("Initial Energy: ", Energy[0])
# print("Final Energy: ", Energy[-1])
# print("Initial Radius: ", Radius[0])
# print("Final Radius: ", Radius[-1])
# print("Number of rejections: ", rejects)