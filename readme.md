# pyclsm

An implementation of a conservative level set method for two-phase flow simulations using the FEniCSx finite element library.

<p align="center">
    <img src="bubble-tutorial.gif"  class="center"/>
</p>


## Usage
### Two-Phase Flow Solver
#### Rising bubble


To illustrate the use of `pyclsm`'s incompressible two-phase flow solver we consider simulating a circular bubble rising due to buoyancy.

##### Domain, Initial and Boundary Conditions 

We will simulate a bubble contained in a $1$ m $\times$ $2$ m cavity with stationary boundaries. To create the mesh we can use the `rectangular_domain` function from the `common` modlue.

```
from common import rectangular_domain
mesh_spacing = 0.05
mesh = rectangular_domain(mesh_spacing, (0, 0), (1, 2))
```

Next we can create the solver. On the creation of the Two-Phase solver we must set the densities and viscosities of each phase $\rho_0, \rho_1, \mu_0, \mu_1$, as well as surface tension $\sigma$ and acceleration due to gravity.

```
from twophase import IncompressibleTwoPhaseFlowSolver
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
```



Here we set the `kinematic` flag to `True` because $\mathbf{u}\cdot\mathbf{n}=0$ (sometimes called the kinematic boundary condition) hold on all the boundaries in our problem.

Next we will set the initial condition on the level set function $\phi$. This sets the initial position of each phase. In areas where $\phi=0$ we will have $\rho=\rho_0, \mu=\mu_0$ and similarly where $\phi=1$ we will have $\rho=\rho_1, \mu=\mu_1$. To set the initial state of $\phi$ we must interpolate the directed values into `solver.level_set.phi` which is a `dolfinx.fem.Function`. The solver class has some methods that will do this for basic shapes such as circle, ellipses and boxes.

```
solver.set_phi_as_circle((0.5, 0.5), 0.25)
```

We will use the *no-slip* conditions on all the stationary walls, i.e. $\mathbf{u}=\mathbf{0}$. Since this is a commonly used boundary condition the solver has a method that will set it for us.

```
solver.set_no_slip_everywhere()
```

##### Time-steping and Visualisation

This solver is now ready to start performing time steps. To perform multiple steps we can use the optional `steps` parameter in the `time_step` method of the two-phase flow solver.

```
T = 2
solver.time_step(math.ceil(T / solver.dt))
```

To visualize the solution we can use the `eval_level_set` and `eval_velocity` methods. Both methods take as inputs one-dimensional arrays of $x$ and $y$ coordinates where the solution will be evaluated.

```
import numpy as np

phase_pts = 250
phase_x, phase_y = np.meshgrid(np.linspace(0, 1, phase_pts), np.linspace(0, 2, phase_pts))
phases = solver.eval_level_set(phase_x.flatten(), phase_y.flatten())

n_vecs = 40
vecs_x, vecs_y = np.meshgrid(np.linspace(0, 1, n_vecs), np.linspace(0, 2, n_vecs))
u, v = solver.eval_velocity(vecs_x.flatten(), vecs_y.flatten())
```

The array `phases` contains the values of $\phi$ at the points in the domain we specified. The interface between the phases can be visualized by plotting the $0.5$-contour of $\phi$.

```
import matplotlib.pyplot as plt
axes = plt.axes()
axes.contour(phase_x, phase_y, phases.reshape((phase_pts, phase_pts)), levels=[0.5], colors=["gray"])
```

We can then visualize the velocity in any way we like.

```
u = u.reshape((n_vecs, n_vecs))
v = v.reshape((n_vecs, n_vecs))
lengths = np.sqrt(np.square(u) + np.square(v))
max_abs = np.max(lengths)
colors = np.array(list(map(plt.cm.cividis, lengths.flatten() / max_abs)))
axes.quiver(vecs_x, vecs_y, u, v, color=colors)

axes.set_aspect("equal")
plt.show()
```









