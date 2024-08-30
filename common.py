from tqdm import tqdm
import math

def step_until(T, solver, method):
    for _ in tqdm(range(math.ceil(T / solver.dt))):
        method(solver)

