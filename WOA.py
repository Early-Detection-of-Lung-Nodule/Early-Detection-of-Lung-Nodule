import numpy as np
import random as rn
import math
import time

# Whale Optimization Algorithm

def WOA(Positions, fobj, VRmin, VRmax, Max_iter):
    N, dim = Positions.shape[0], Positions.shape[1]
    lb = VRmin[1, :]
    ub = VRmax[1, :]
    Leader_pos = np.zeros((dim, 1))
    Leader_score = float('inf')

    fitness = fobj(Positions)

    Convergence_curve = np.zeros((Max_iter, 1))
    t = 0
    ct = time.time()
    while t < Max_iter:
        print(t)
        for i in range(N):
            # Return back the search agents that go beyond the boundaries of the search space
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb

            #  Update the leader
            if fitness[i] < Leader_score:
                Leader_score = fitness[i]  # Update alpha
                Leader_pos = Positions[i]

        a = 2 - t * ((2) / Max_iter)  # a decreases linearly fron 2 to 0 in Eq. (2.3)

        # a2 linearly dicreases from -1 to -2 to calculate t in Eq. (3.12)
        a2 = -1 + t * ((-1) / Max_iter)

        # Update the Position of search agents
        for i in range(N):
            r1 = rn.random()
            r2 = rn.random()

            A = 2 * a * r1 - a
            C = 2 * r2

            b = 1
            l = (a2 - 1) * rn.random() + 1

            p = rn.random()

            for j in range(dim):
                if p < 0.5:
                    if abs(A) >= 1:
                        rand_leader_index = math.floor(N * rn.random() + 1)
                        X_rand = Positions[rand_leader_index - 1, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand
                    elif abs(A) < 1:
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * D_Leader
                elif p >= 0.5:
                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    Positions[i, j] = distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi) + Leader_pos[j]
        Convergence_curve[t] = Leader_score
        t = t + 1
    Leader_score = Convergence_curve[Max_iter - 1]
    ct = time.time() - ct

    return Leader_score, Convergence_curve, Leader_pos, ct


