import numpy as np
import random
import time


# Particle Swarm Optimization Algorithm (PSO)
def PSO(X, fitness_function, minx, maxx, max_iter):
    # Set the constants
    w = 0.5  # inertia weight
    c1 = 2.0  # cognitive coefficient
    c2 = 2.0  # social coefficient
    dim = X.shape[1]  # The number of variables in each particle (Chlen)

    # Initialize variables for best positions and fitness values
    Npop = X.shape[0]  # Number of particles
    bestPos = X.copy()  # Best positions for each particle (initially their own positions)
    bestFitness = np.full(Npop, float('inf'))  # Best fitness for each particle
    velocity = np.zeros_like(X)  # Initialize velocity to zero

    best_pos_swarm = np.zeros(dim)  # Global best position of the swarm
    best_fitness_swarm = float('inf')

    convergence = np.zeros(max_iter)
    start_time = time.time()

    # Iterate over the max_iter
    for t in range(max_iter):
        for i in range(Npop):
            particle = X[i]
            fitness = fitness_function(particle)

            # Update the best position of the particle
            if fitness < bestFitness[i]:
                bestFitness[i] = fitness
                bestPos[i] = particle.copy()

            # Update the global best position of the swarm
            if fitness < best_fitness_swarm:
                best_fitness_swarm = fitness
                best_pos_swarm = particle.copy()

            # Update velocity
            r1 = random.random()
            r2 = random.random()
            velocity[i] = w * velocity[i] + c1 * r1 * (bestPos[i] - particle) + c2 * r2 * (best_pos_swarm - particle)

            # Update position
            X[i] = particle + velocity[i]

            # Clip the position to be within the bounds
            X[i] = np.clip(X[i], minx[i], maxx[i])
        convergence[t] = best_fitness_swarm
    elapsed_time = time.time() - start_time
    return best_fitness_swarm, convergence, best_pos_swarm, elapsed_time

