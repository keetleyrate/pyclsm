import dolfinx
import dolfinx.fem.petsc
import ufl
from ufl import inner, dx, grad
from normal import NormalProjector
from ellipticproject import EllipticProjector
from scipy.integrate import simpson
import numpy as np
import basix
import gmsh
from common.visualize import eval_sol
import sys
from mpi4py import MPI
from common.common import interpolate_expression, interpolate_from_old_mesh
import copy
from common.visualize import * 

def normalize_and_threshold_dist(d, d_min, d_max):
    return (np.maximum(d_min, np.minimum(d, d_max)) - d_min) / (d_max - d_min)


def create_spaces_and_functions(mesh, p):
    scalar_element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), p)
    vector_element = basix.ufl.element("Lagrange", mesh.topology.cell_name(), p, shape=(mesh.geometry.dim,))
    scalar_space = dolfinx.fem.functionspace(mesh, scalar_element)
    vector_space = dolfinx.fem.functionspace(mesh, vector_element)
    ϕ = dolfinx.fem.Function(scalar_space)
    ϵ = dolfinx.fem.Function(scalar_space)
    κ = dolfinx.fem.Function(scalar_space)
    n = dolfinx.fem.Function(vector_space)
    h = dolfinx.fem.Function(scalar_space)
    grad_ϕ = dolfinx.fem.Function(vector_space)
    return scalar_space, vector_space, ϕ, ϵ, κ, n, h, grad_ϕ




class ConservativeLevelSet:

    def __init__(self, domain, h_min, dt, p=1, reinit_iters=7, solver_options=None | dict) -> None:
        self.p = p
        self.dt = dt
        self.h_min = h_min
        self.reinit_iters = reinit_iters
        self.domain = domain
        self.dτ = 0.05
        self.scalar_space, self.vector_space, self.ϕ, self.ϵ, self.κ, self.n, self.h, self.grad_ϕ = create_spaces_and_functions(domain.mesh, p)
        self.ϵ = dolfinx.fem.Constant(self.scalar_space.mesh, h_min / 2)
        self.advection_bcs = []
        self.reinit_bcs = []
        self.n = dolfinx.fem.Function(self.vector_space)
        self.options = solver_options
        self.m = 100


    def create_test_and_trail_functions(self):
        self.trial = ufl.TrialFunction(self.ϕ.function_space)
        self.test = ufl.TestFunction(self.ϕ.function_space)

    def set_interface_bc(self, tol=0.05):
        indices = np.flatnonzero(np.abs(self.ϕ.x.array - 0.5) < tol)
        bc = dolfinx.fem.dirichletbc(self.ϕ, indices)
        self.reinit_bcs = [bc]

    def compute_curvature(self):
        assert self.normal_projector is not None
        self.normal_projector.compute_normals()
        n = self.normal_projector.n
        self.curvature_projector = EllipticProjector(0)
        self.curvature_projector.build_problem(self.ϕ.function_space, -ufl.div(n), self.h_min)
        self.curvature_projector.project()
        return self.curvature_projector.solution
    
    def compute_normals(self):
        α = 1e-4
        β = 10
        trail = ufl.TrialFunction(self.vector_space)
        test = ufl.TestFunction(self.vector_space)
        g = ufl.grad(self.ϕ)
        ϵ = self.ϵ
        m = ϵ * g / ufl.sqrt(ϵ*ϵ*ufl.inner(g, g) + α*α*ufl.exp(-β*ϵ*ϵ*ufl.inner(g, g)))
        
        lhs = ufl.inner(trail, test) * ufl.dx
        rhs = ufl.inner(m, test) * ufl.dx
        bc_val_x0 = np.array([0.0, -1.0])
        bc_val_y0 = np.array([-1.0, 0.0])
        zero_bc_val = np.array([0.0, 0.0])
        domain = self.domain.mesh
        facets_x0 = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, 
                                                lambda x: np.isclose(x[0], 0))
        facets_y0 =dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, 
                                                lambda x: np.isclose(x[1], 0))
        facets_x1 = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, 
                                                lambda x: np.isclose(x[0], 1))
        facets_y1 =dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, 
                                                lambda x: np.isclose(x[1], 1))
        dofs_x0 = dolfinx.fem.locate_dofs_topological(self.n.function_space, domain.topology.dim-1, facets_x0)
        dofs_y0 = dolfinx.fem.locate_dofs_topological(self.n.function_space, domain.topology.dim-1, facets_y0)
        dofs_x1 = dolfinx.fem.locate_dofs_topological(self.n.function_space, domain.topology.dim-1, facets_x1)
        dofs_y1 = dolfinx.fem.locate_dofs_topological(self.n.function_space, domain.topology.dim-1, facets_y1)
        bc_x0 = dolfinx.fem.dirichletbc(bc_val_x0, dofs_x0, self.n.function_space)
        bc_y0 = dolfinx.fem.dirichletbc(bc_val_y0, dofs_y0, self.n.function_space)
        bc_x1 = dolfinx.fem.dirichletbc(zero_bc_val, dofs_x1, self.n.function_space)
        bc_y1 = dolfinx.fem.dirichletbc(zero_bc_val, dofs_y1, self.n.function_space)
        problem = dolfinx.fem.petsc.LinearProblem(
            dolfinx.fem.form(lhs),
            dolfinx.fem.form(rhs),
            bcs=[bc_y0, bc_x0, bc_x1, bc_y1],
            u=self.n,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre"},
            petsc_options_prefix="normal_"
        )
        problem.solve()


    def build_gls_advection_problem(self, u: dolfinx.fem.Function):
        trail, test = self.trial, self.test

        dt = self.dt
        ϕ = self.ϕ
        c = 1
        δ = c * ufl.CellDiameter(ϕ.function_space.mesh) / (2 * ufl.sqrt(ufl.inner(u, u)))
        def L(ϕ):
            return 1/dt * ϕ + 1/2 * ufl.div(ϕ * u)
        f = 1/dt * ϕ - 1/2 * ufl.div(ϕ * u)
        lhs = L(trail) * test * ufl.dx + δ * L(trail) * L(test) * ufl.dx
        rhs = f * test * ufl.dx + δ * f * L(test) * ufl.dx
        self.gls_advection_problem = dolfinx.fem.petsc.LinearProblem(
            dolfinx.fem.form(lhs),
            dolfinx.fem.form(rhs),
            bcs=self.advection_bcs,
            u=self.ϕ,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre"},
            petsc_options_prefix="gls_advection_"
        )


    def build_reinit_problem(self, use_mesh_eps=True):
        dτ = self.dτ
        ϕ = ufl.min_value(ufl.max_value(self.ϕ, 0.0), 1.0)
        trial, test = self.trial, self.test
        if use_mesh_eps:
            ϵ = ufl.CellDiameter(self.ϕ.function_space.mesh) / 2
        else:
            ϵ = self.ϵ
        n = self.n
        form = (
            1/dτ*(trial-ϕ)*test
            +1/2*ufl.div((trial+ϕ-2*trial*ϕ)*n)*test
            +ϵ/2*ufl.inner(ufl.grad(trial+ϕ),n)*ufl.inner(ufl.grad(test), n)
            +ϵ/2*ufl.inner((1-ufl.inner(n,n))*ufl.grad(trial+ϕ), ufl.grad(test))
        ) * ufl.dx

        rhs = dolfinx.fem.form(ufl.rhs(form))
        lhs = dolfinx.fem.form(ufl.lhs(form))
        self.reinit_problem = dolfinx.fem.petsc.LinearProblem(
            lhs,
            rhs,
            bcs=self.reinit_bcs,
            u=self.ϕ,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre"},
            petsc_options_prefix="reinit_"
        )


    
    def build_problems(self, u):
        self.create_test_and_trail_functions()
        # self.normal_projector = NormalProjector(self.ϕ, self.h_min, self.p, self.c_normal)
        self.normal_projector.build_problem()
        self.build_gls_advection_problem(u)
        self.gls_advection_problem.solve()
        self.build_reinit_problem()


    def advect(self, u, show=True):
        self.create_test_and_trail_functions()
        self.build_gls_advection_problem(u)
        self.gls_advection_problem.solve()
        self.compute_normals()
        self.build_reinit_problem()
        last = dolfinx.fem.Function(solver.scalar_space)
        last.x.array[:] = solver.ϕ.x.array
        for _ in range(self.reinit_iters):
            self.reinit_problem.solve()
            res = dolfinx.fem.form((phi_m - solver.ϕ) * (phi_m - solver.ϕ) * ufl.dx)
            res = dolfinx.fem.assemble_scalar(res)
            phi_m.x.array[:] = solver.ϕ.x.array
            if res < 1e-4:
                print(f"converged after {i + 1} iterations")
                break
            if show:
                print("REINIT Residual:", res)


        
    

    def compute_h(self):
        interpolate_expression(ufl.CellDiameter(self.domain.mesh), self.h)
        #interpolate_expression(self.h**(1 - self.d) / 2, self.ϵ)
  

    # def set_phi_from_sdf(self, sdf, load_as=None):
    #     self.compute_h()
    #     self.dist = dolfinx.fem.Function(self.scalar_space)
    #     fem_func = fem_function_from_sdf(sdf, load_as)
    #     self.dist.interpolate(lambda x: fem_func(x))
    #     interpolate_expression(1 / (1 + ufl.exp(self.dist / self.ϵ)), self.ϕ)



class AdaptiveLevelSetDomain:

    def __init__(self, inital_mesh: dolfinx.mesh.Mesh, h_min, h_max, m=10):
        """inital_mesh: A dolfinx mesh object of the inital domain.
           f: A function f: R^n -> R whose zero level set is the inital interface.
              Must satisfy the condtion that f > 0 in the interior of the zero level set
              and f < 0 outside the interior.
           g: A function which returns the gradient of f.
           h_min: Minimum mesh size.
           h_max: Maximum mesh size.
        """
        self.mesh = inital_mesh
        self.tree = dolfinx.geometry.bb_tree(inital_mesh, 2)
        self.h_min = h_min
        self.h_max = h_max
        self.m = m
        

    def compute_grad_norm(self, ϕ):
        norm_grad_ϕ = dolfinx.fem.Function(ϕ.function_space)
        grad_norm_expr = ufl.sqrt(ufl.inner(ufl.grad(ϕ), ufl.grad(ϕ)))
        interpolate_expression(grad_norm_expr, norm_grad_ϕ)
        return norm_grad_ϕ

    def remesh(self, solver: ConservativeLevelSet, interface_tol=0.2, sample_rate=1, show=False):
        self.mesh.comm.Barrier()
        comm = self.mesh.comm
        local_num_dofs = solver.ϕ.function_space.dofmap.index_map.size_local
        local_phi = solver.ϕ.x.array[:local_num_dofs]
        dof_coords = solver.ϕ.function_space.tabulate_dof_coordinates()[:local_num_dofs]
        phi_values = comm.gather(local_phi, root=0)
        coords = comm.gather(dof_coords, root=0)
        gmsh.initialize()
        if self.mesh.comm.rank == 0:
            # Genourate the new mesh
            gmsh.model.add("refined_mesh")
            phi_values = np.concatenate(phi_values)
            #coords = np.concatenate(coords)
            # p = 0.02
            # THIS IS A GOOD, extect if phi is not in [0, 1] the remeshing indicator doesnt behave well, i.e nans
            # basically means reinitalatins isnt working properly.
            # ind = np.maximum(4 * phi_values * (1 - phi_values), 0)


            #cell_area = self.h_min + (self.h_max - self.h_min) * (1 - ind ** p)
            #dist = np.abs(self.h_min / 2 * np.log(np.abs((1 - phi_values) / phi_values)))
            #dist /= np.max(dist)
            #t = normalize_and_threshold_dist(dist, 2 * self.h_min, 10 *  self.h_min)
            #print(np.min(t), np.max(t))
            #cell_area = self.h_min * (1 - t) + t * self.h_max

            #view_data = np.column_stack((coords, cell_area)).flatten()
            #h_view = gmsh.view.add("hView")
            #gmsh.view.add_list_data(h_view, "SP", len(coords), view_data)

            #h_field = gmsh.model.mesh.field.add("PostView")
            #gmsh.model.mesh.field.setNumber(h_field, "ViewIndex", 0)



            coords = np.concatenate(coords)
            interface_inds = np.abs(phi_values - 0.5) < interface_tol
            interface_point_tags = [gmsh.model.occ.add_point(*p) for p in coords[interface_inds][::sample_rate]]


            
            surf = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2, [surf], name="")

            distance_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.set_numbers(distance_field, "PointsList", interface_point_tags)

            threshold_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.set_number(threshold_field, "InField", distance_field)
            gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", self.h_min)
            gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", self.h_max)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 3 * self.h_min)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 20 * self.h_min)

            gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    
            gmsh.model.mesh.generate(2)
            if show:
                gmsh.fltk.run()

        new_mesh_data = dolfinx.io.gmsh.model_to_mesh(
            gmsh.model, 
            self.mesh.comm, 
            rank=0, 
            gdim=2
        )
        new_mesh = new_mesh_data.mesh
        gmsh.finalize()
        # Create new function spaces and functions
        scalar_space, vector_space, ϕ, ϵ, κ, n, h, grad_ϕ = create_spaces_and_functions(new_mesh, solver.p)
        # Interpolate old ϕ into new mesh
        interpolate_from_old_mesh(solver.ϕ, ϕ, self.tree)
        ϵ = dolfinx.fem.Constant(scalar_space.mesh, self.h_min / 2)
        # Update spaces and functions of solver
        solver.scalar_space, solver.vector_space, solver.ϕ, solver.ϵ, solver.κ, solver.n, solver.h, solver.grad_ϕ = scalar_space, vector_space, ϕ, ϵ, κ, n, h, grad_ϕ
        # Update the mesh and tree of the domain
        self.mesh = new_mesh
        solver.ϕ.name = "phi"
        self.tree = dolfinx.geometry.bb_tree(new_mesh, 2)

       



   


        

        

            
