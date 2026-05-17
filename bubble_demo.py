from common.domains import rectangular_domain

mesh_spacing = 0.05/2
mesh, tree = rectangular_domain(mesh_spacing, (0, 0), (1, 2))
from twophaseflow import IncompressibleTwoPhaseFlowSolver
solver = IncompressibleTwoPhaseFlowSolver(
    mesh=mesh,
    h=mesh_spacing,
    dt=0.5*mesh_spacing,
    rho0=1,
    rho1=0.1,
    mu0=0.01,
    mu1=0.001,
    sigma=0.1,
    g=0.98,
    kinematic=True
)

solver.set_phi_as_circle((0.5, 0.5), 0.25)
solver.set_no_slip_everywhere()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from common.visualize import *
import matplotx

plt.style.use(matplotx.styles.ayu["mirage"])

def update(frame, axes, solver):
    for ax in axes:
        ax.clear()
    fem_plot_contor(fig, axes[1], solver.level_set.phi, mesh, tree, (0, 1), (0, 2), 200, levels=[0.5], colors=["gray"], linewidths=[0.5])
    fem_plot_vectors(axes[1], solver.u, mesh, tree, (0, 1), (0, 2), 35)
    fem_plot_contor_filled(fig, axes[0], solver.level_set.phi, mesh, tree, (0, 1), (0, 2), 200, levels=250)
    fem_plot_contor_filled(fig, axes[2], solver.velocity_magniute(), mesh, tree, (0, 1), (0, 2), 200, 250)

    for ax in axes:
        ax.set_xlabel("$x$")
        ax.set_xlabel("$y$")
    
    axes[0].set_title(f"$\phi,\quad t={solver.t:.2f}$")
    axes[1].set_title(r"$\mathbf{u},\quad" + f" t={solver.t:.2f}$")
    axes[2].set_title(r"$||\mathbf{u}||,\quad" f" t={solver.t:.2f}$")
    for ax in axes:
        ax.set_aspect("equal")
    plt.tight_layout()
    solver.time_step()
    print(f"{solver.t / 2:.3f}")









T = 5
# solver.time_step(math.ceil(T / solver.dt))

fig, axes = plt.subplots(1, 3)
ani = animation.FuncAnimation(fig, update, math.ceil(T / solver.dt), fargs=(axes, solver))
ani.save("bubble-tutorial-long.gif", writer="Pillow", fps=30)


# n_phase_pts = 250
# phi_x, phi_y = np.meshgrid(np.linspace(0, 1, n_phase_pts), np.linspace(0, 2, n_phase_pts))
# phi = solver.eval_level_set(phi_x.flatten(), phi_y.flatten())

# axes.contour(phi_x, phi_y, phi.reshape((n_phase_pts, n_phase_pts)), levels=[0.5], colors=["gray"])


# n_vecs = 40
# vecs_x, vecs_y = np.meshgrid(np.linspace(0, 1, n_vecs), np.linspace(0, 2, n_vecs))
# u, v = solver.eval_velocity(vecs_x.flatten(), vecs_y.flatten())

# u = u.reshape((n_vecs, n_vecs))
# v = v.reshape((n_vecs, n_vecs))
# lengths = np.sqrt(np.square(u) + np.square(v))
# max_abs = np.max(lengths)
# colors = np.array(list(map(plt.cm.cividis, lengths.flatten() / max_abs)))
# axes.quiver(vecs_x, vecs_y, u, v, color=colors)

# axes.set_aspect("equal")
# plt.show()