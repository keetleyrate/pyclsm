# pyclsm

An implementation of a conservative level set method for two-phase flow simulations using the FEniCSx finite element library.

## Usage

### Navier-Stokes Projection Solver

#### Couette flow

To demonstrate how to use the pyclsm's fluid solver we will consider the problem of 2D Couette flow. In Couette flow, viscous fluid occupies the space between two infinitely long surfaces, one of which moves tangentially relative to the other. We assume that the velocity ($\mathbf{u}=(u,v)$) is unidirectional so that $v=0$. Incompressiblity then implies that $\frac{∂ u}{∂ x} = 0$ or equivalently that $u$ only depends on $y$ and $t$.


<p align="center">
    <img src="couette.png" width="250" height="250"  class="center"/>
</p>

For simplicity, we will consider when the top surface is moving at a constant velocity of $1$ meter per second, the plates are $1$ meter apart and furthermore we have $\rho=1, \mu=1$. In this case the solution is known to be:

$u(y, t)=y - \frac{1}{\pi}\sum\limits^\infty_{n=1}\frac{1}{n}e^{-n^2\pi^2}\sin\left(1-y\right).$

Let us try to recover the analytical solution using pyclsm's fluid Navier-Stokes solver.

##### Domain and Boundary Conditions

First we need to create a domain to solve the problem in. Since `pyclsm` uses the `dolfinx` package to solve PDE's this must be in the form of a `dolfinx.mesh.Mesh` object. The `common` module provides some functions to create simple rectangular meshes. The following creates a mesh of $[0, 1] \times [0, 1]$ with square finite elements.

```
from common import unit_square
mesh_spacing = 0.05
mesh = unit_square(mesh_spacing)
```

The `mesh_spacing` parameter controls the spacing between elements ($h$ or $\Delta x$) meaning we will find the solution on a $20\times20$ grid. Next we can create the fluid solver.

```
from navierstokes import IncompressibleNavierStokesSolver
time_step = 0.0005
solver = IncompressibleNavierStokesSolver(mesh, time_step)
solver.set_density_as_const(1)
solver.set_viscosity_as_const(1)
```

We must pass both the mesh and the time step we wish to use ($k$ or $\Delta t$). We also tell the solver to set $\rho=1$ and $\mu=1$ over the entire domain as on creation they will be zero everywhere. Next we can set all our boundary conditions.

```
from common import x_equals, y_equals
solver.set_velocity_bc(y_equals(0), (0, 0))
solver.set_velocity_bc(y_equals(1), (1, 0))
solver.set_y_velocity(x_equals(0), 0)
solver.set_y_velocity(x_equals(1), 0)
```

Each of the previous methods takes two arguments: a function which when given an array of the mesh points returns an array of booleans which are `True` corresponding to mesh points where the boundary condition is to be set. The common module provides some useful functions that can be used here for rectangular domains. The second arguments are the value of the solution on that particular boundary. In the above code it appears as if we have not done anything about the boundary condition in which $\frac{∂ u}{∂ x}=0$. However, by default the solver assumes that *any boundaries that do not have Dirichlet boundary conditions have natural boundary conditions on velocity*. That is $∇\mathbf{u}\cdot\mathbf{n}=0$ on those boundaries. In our problem since $v=0$ this condition reduces to $\frac{∂ u}{∂ x}=0$. Finally, now that we have dealt with all our boundary conditions, we can tell the solver we are ready to start solving the problem.

```
solver.reset()
```

Anytime the boundary conditions of the solver are changed the `reset` method must be called again.

##### Time-steping and Visualisation

To perform a single time step of the problem we can now call the `time_step` method of the fluid solver.

```
import math
T = 0.5
for _ in range(math.ceil(T / time_step)):
    solver.time_step()
```

At anytime we can get the solver evaluate the finite element solution at any points in the domain.

```
import numpy as np 
y = np.linspace(0, 1, 150)
x = 0.5 * np.ones(150)
u, v = solver.eval_velocity(x, y)
```

This allows us to visualise the solution with a plotting library of our choice. For example, with `matplotlib`.

```
import matplotlib.pyplot as plt 
plt.plot(u, y)
plt.show()
```

Now we can compare the numerical solution to the exact over time.

<p align="center">
    <img src="couette-tutorial.gif" width="500"  class="center"/>
</p>

### Two-Phase Flow Solver

#### Rising bubble

To illustrate the use of `pyclsm`'s incompressible two-phase flow solver we consider simulating a circular bubble rising due to buoyancy.

##### Domain, Initial and Boundary Conditions 

We will simulate a bubble contained in a $1$m $\times$ $2$m cavity with stationary boundaries. To create the mesh we can use the `rectangular_domain` function from the `common` modlue.

```
from common import rectangular_domain
mesh_spacing = 0.05
mesh = rectangular_domain(mesh_spacing, (0, 0), (1, 2))
```

Next we can create the solver. On the creation of the Two-Phase solver we must set the densities and viscosities of each phase \(\rho_0, \rho_1, \mu_0, \mu_1\), as well as surface tension \(σ\) and acceleration due to gravity.

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

Next we will set the initial condition on the level set function \(\phi\). This sets the initial position of each phase. In areas where \(\phi=0\) we will have \(\rho=\rho_0, \mu=\mu_0\) and similarly where \(\phi=1\) we will have \(\rho=\rho_1, \mu=\mu_1\). To set the initial state of \(\phi\) we must interpolate the directed values into `solver.level_set.phi` which is a `dolfinx.fem.Function`. The solver class has some methods that will do this for basic shapes such as circle, ellipses and boxes.

```
solver.set_phi_as_circle((0.5, 0.5), 0.25)
```

We will use the *no-slip* conditions on all the stationary walls, i.e. \(\mathbf{u}=\mathbf{0}\). Since this is a commonly used boundary condition the solver has a method that will set it for us.

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

<p align="center">
    <img src="bubble-tutorial.gif" width="500"  class="center"/>
</p>











