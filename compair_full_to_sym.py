import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("sym_data.pkl", "rb") as infile:
    sym = pickle.load(infile)
with open("full_data.pkl", "rb") as infile:
    full = pickle.load(infile)
fig, ax = plt.subplots(2, 2)
ax[0, 0].set_xlabel("$x$")
ax[0, 0].set_ylabel("$u$")
ax[0, 1].set_xlabel("$x$")
ax[0, 1].set_ylabel("$v$")
ax[1, 0].set_xlabel("$x$")
ax[1, 0].set_ylabel(r"$\phi$")
ax[1, 1].set_xlabel("$x$")
ax[1, 1].set_ylabel(r"$\kappa$")
dt = 0.01
x = np.linspace(0, 2, 250)
for i in [99]:
    for j, file in enumerate([sym, full]):
        us = file[i]["u"]
        vs = file[i]["v"]
        phi = file[i]["phi"]
        ks = file[i]["k"]
        # x, y, u, v = file[i]["vecs"]
        # if j == 0:
        #     ax[j].quiver(x, y, u, v)
        # else:
        #     n = len(x)//2
        #     ax[j].quiver(x[n:, n:], y[n:, n:], u[n:, n:], v[n:, n:])
        ax[0, 0].plot(x, us, label=("full" if j == 1 else "sym"), linestyle=("-" if j == 0 else "--"), color="black")
        ax[0, 1].plot(x, vs, label=("full" if j == 1 else "sym"), linestyle=("-" if j == 0 else "--"), color="black")
        ax[1, 0].plot(x, phi, label=("full" if j == 1 else "sym"), linestyle=("-" if j == 0 else "--"), color="black")
        ax[1, 1].plot(x, ks, label=("full" if j == 1 else "sym"), linestyle=("-" if j == 0 else "--"), color="black")
ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()
# ax[0].set_title("Sym")
# ax[1].set_title("Full")
plt.show()