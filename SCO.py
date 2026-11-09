import numpy as np
import time

# Single Candidate Optimization Algorithm

def SCO(X,obj_func, lb, ub, max_iter):
    dim, pop_size = X.shape

    # --- Initialize ---
    t = 0  # iteration counter
    a = 0  # phase flag
    p = 0  # progress flag
    m = 5  # switch counter threshold
    c = 0  # phase control counter

    b = 1.5  # constant used in w(t)
    lb, ub = np.array(lb), np.array(ub)

    # Initialize candidate solutions (Eq. 7)
    lb = np.array(lb).flatten()[:dim]
    ub = np.array(ub).flatten()[:dim]
    X = lb + np.random.rand(pop_size, dim) * (ub - lb)
    fitness = np.array([obj_func(x) for x in X])

    # Determine the best solution
    best_idx = np.argmin(fitness)
    gbest = X[best_idx].copy().flatten()

    f_gbest = fitness[best_idx]
    convergence = np.zeros((max_iter))
    start_time = time.time()
    # --- Main Loop ---
    while t < max_iter:
        for i in range(pop_size):
            x = X[i].copy()
            new_x = np.zeros(dim)

            # Phase 1 or 2 selection
            if a < m:
                # Eq. (2): Exploration
                for j in range(dim):
                    r1 = np.random.rand()
                    if r1 < 0.5:
                        new_x[j] = gbest[j] + np.random.rand() * abs(gbest[j])
                    else:
                        new_x[j] = gbest[j] - np.random.rand() * abs(gbest[j])
            else:
                # Eq. (4): Exploitation (deep search)
                w_t = np.exp(-((t / max_iter) ** b))
                for j in range(dim):
                    r2 = np.random.rand()
                    if r2 < 0.5:
                        new_x[j] = gbest[j] + (r2 * w_t * (ub[j] - lb[j]))
                    else:
                        new_x[j] = gbest[j] - (r2 * w_t * (ub[j] - lb[j]))

            # Boundary check (Eq. 6)
            new_x = np.clip(new_x, lb, ub)

            # Evaluate new candidate
            f_new = obj_func(new_x)

            # Update global best (Steps 17–20)
            if f_new < f_gbest:
                gbest = new_x.copy()
                f_gbest = f_new
                p = 1
            else:
                p = 0

            X[i] = new_x

        # Phase switching logic (Steps 7–12)
        if p == 0:
            c += 1
            if c == m:
                c = 0
                a += 1  # switch to deep search
        else:
            a = 0

        convergence[t] = float(f_gbest)
        t += 1

    elapsed_time = start_time- time.time()
    return gbest, convergence, f_gbest, elapsed_time

