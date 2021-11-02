import random
import numpy as np
from pulp import *


def prob_gen(minL=50, maxL=100, minN=20, maxN=30, maxA=70, maxB=20, maxD=150, maxC=20):
    '''
    minL, maxL: the minimum and maximum length of the berth sections;
    minN, maxN: the minimum and maximum number of the vessel number;
    maxA: the maximum arrival time of a vessel;
    maxB: the maximum operation time of a vessel;
    maxD: the maximum departure time of a vessel;
    maxC: the maximum cost for both additional travel and delayed departure.
    '''
    L = random.uniform(minL, maxL)
    n = random.uniform(minN, maxN)

    vessels = []
    for _ in range(n):
        a = random.uniform(0, maxA)
        b = random.uniform(0, maxB)
        d = random.uniform(a + b, maxD)
        p = random.uniform(0, L)
        l = random.uniform(0, L)
        c1 = random.uniform(0, maxC)
        c2 = random.uniform(0, maxC)
        vessels.append((p, a, b, d, l, c1, c2))
    return {'L': L, 'N': n, 'v': vessels}


params = prob_gen()
L, N, M = params['L'], params['N'], 1e6
a, b, d, p, l, c1, c2 = zip(*params['v'])

# construct the problem in pulp
prob = LpProblem("The Berth Allocation", LpMinimize)

abs_x_minus_p = [pulp.LpVariable(f'abs_x_minus_p_{i}', lowBound=0, cat="Continuous") for i in range(N)]
pos_y_plus_b_minus_d = [pulp.LpVariable(f'pos_y_plus_b_minus_d_{i}', lowBound=0, cat="Continuous") for i in range(N)]
x = [pulp.LpVariable(f'x_{i}', lowBound=0, upBound=L - l[i], cat="Continuous") for i in range(N)]
y = [pulp.LpVariable(f'y_{i}', lowBound=a[i], cat="Continuous") for i in range(N)]
zx = [pulp.LpVariable(f'zx_{ij // N}_{ij % N}', lowBound=0, upBound=1, cat="Integer") for ij in range(N ** 2)]
zy = [pulp.LpVariable(f'zy_{ij // N}_{ij % N}', lowBound=0, upBound=1, cat="Integer") for ij in range(N ** 2)]

# objective function
prob += pulp.lpSum([c1[n] * abs_x_minus_p[n] for n in range(N)] + [c2[n] * pos_y_plus_b_minus_d[n] for n in range(N)])

# add constraints
for n in range(N):
    prob += x[n] - p[n] <= abs_x_minus_p[n], f"Available check for abs_x_minus_p_{n}-"
    prob += p[n] - x[n] <= abs_x_minus_p[n], f"Available check for abs_x_minux_p_{n}+"

for n in range(N):
    prob += y[n] + b[n] - d[n] <= pos_y_plus_b_minus_d[n], f"Available check for pos_y_plus_b_minus_d_{n}"

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        prob += x[i] + l[i] - x[j] + M * zx[i * N + j] <= M, f"Avialable check for x_{i} between x_{j}"
        prob += y[i] + b[i] - y[j] + M * zy[i * N + j] <= M, f"Avialable check for y_{i} between y_{j}"
        if i > j:
            continue
        prob += zx[i * N + j] + zx[i + j * N] + zy[i * N + j] + zy[
            i + j * N] >= 1, f"Available check for rectangle with {i}{j})"

# solve the problem
status = prob.solve()
print(pulp.LpStatus[status])
