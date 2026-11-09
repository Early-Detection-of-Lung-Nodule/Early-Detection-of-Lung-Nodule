import numpy as np
import time

# Genghis Khan Shark Optimizer

def GKSO(X, ObjectiveFunction, lb, ub, T):
    dim, N = X.shape
    lb = np.asarray(lb).flatten()
    ub = np.asarray(ub).flatten()

    # --- Step 1: Initialization ---
    lb = np.tile(lb, (N, 1))
    ub = np.tile(ub, (N, 1))

    fitness = np.array([ObjectiveFunction(x) for x in X])
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    F_best = fitness[best_idx]

    Convergence_curve = []

    # --- Step 2: Main Loop ---
    start_time = time.time()
    for t in range(T):

        # Stage 1: Moving Toward the Best Hunting Position (Exploitation)
        for i in range(N):
            s = np.exp(-fitness[i])  # Olfactory intensity
            r = np.random.rand()

            if r < 0.5:
                # Exploration (return to wandering)
                X[i] = X[i]
            else:
                # Exploitation (move towards best)
                step = s * (X_best - X[i]) + np.random.randn(dim) * 0.1
                X[i] = X[i] + step

        # Stage 2: Parabolic Foraging
        for i in range(N):
            a = np.random.uniform(-1, 1, dim)
            parabola = a * (X_best - X[i]) ** 2
            X[i] = X[i] + parabola * np.random.rand()

        # Stage 3: Self-Protection Mechanism
        for i in range(N):
            if np.random.rand() < 0.3:
                # Local mutation around best
                mutation = X_best + np.random.randn(dim) * 0.1 * (ub - lb)
                if ObjectiveFunction(mutation) < fitness[i]:
                    X[i] = mutation

        # --- Step 4: Boundary Check ---
        X = np.clip(X, lb, ub)

        # --- Step 5: Evaluate New Fitness ---
        fitness = np.array([ObjectiveFunction(x) for x in X])
        min_idx = np.argmin(fitness)

        # Update global best
        if fitness[min_idx] < F_best:
            X_best = X[min_idx].copy()
            F_best = fitness[min_idx]

    Convergence_curve.append(F_best)
    elapsed_time = time.time() - start_time

    return F_best, Convergence_curve, X_best, elapsed_time



