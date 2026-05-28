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


    def __init__(self, domain, h_min, dt, dτ=0.05, p=1, c_kappa=2, reinit_tol=1e-5, sym_bcs=False, max_reinit_iters=100, solver_options=None | dict) -> None:
        self.p = p
        self.dt = dt
        self.h_min = h_min
        self.Ck = c_kappa
        self.max_reinit_iters = max_reinit_iters
        self.reinit_tol = reinit_tol
        self.domain = domain
        self.dτ = dτ
        self.scalar_space, self.vector_space, self.ϕ, self.ϵ, self.κ, self.n, self.h, self.grad_ϕ = create_spaces_and_functions(domain.mesh, p)
        self.ϵ = dolfinx.fem.Constant(self.scalar_space.mesh, 2 * h_min)
        self.advection_bcs = []
        self.reinit_bcs = []
        self.n = dolfinx.fem.Function(self.vector_space)
        self.phi_temp = dolfinx.fem.Function(self.scalar_space)
        self.options = solver_options
        self.sym_bcs = sym_bcs
        self.m = 100


    def create_test_and_trail_functions(self):
        self.trial = ufl.TrialFunction(self.ϕ.function_space)
        self.test = ufl.TestFunction(self.ϕ.function_space)

    def set_interface_bc(self, tol=0.05):
        indices = np.flatnonzero(np.abs(self.ϕ.x.array - 0.5) < tol)
        bc = dolfinx.fem.dirichletbc(self.ϕ, indices)
        self.reinit_bcs = [bc]

    def compute_curvature(self):
        g = self.grad_ϕ
        gnorm = ufl.sqrt(ufl.inner(g, g) + 1e-3)
        trail = ufl.TrialFunction(self.scalar_space)
        test = ufl.TestFunction(self.scalar_space)
        form = (trail * test - ufl.inner(g/gnorm, ufl.grad(test)) + self.Ck * self.h_min * self.h_min * ufl.inner(ufl.grad(trail), ufl.grad(test))) * ufl.dx

        lhs = dolfinx.fem.form(ufl.lhs(form))
        rhs = dolfinx.fem.form(ufl.rhs(form))
        problem = dolfinx.fem.petsc.LinearProblem(
            lhs,
            rhs,
            bcs=[],
            u=self.κ,
            petsc_options={"ksp_type": "minres", "pc_type": "hypre"},
            petsc_options_prefix="curvature_"
        )
        problem.solve()
    
    def compute_gradient(self):
        trail = ufl.TrialFunction(self.vector_space)
        test = ufl.TestFunction(self.vector_space)
        form = (ufl.inner(trail, test) - ufl.inner(ufl.grad(self.ϕ), test) + self.Ck * self.h_min * self.h_min * ufl.inner(ufl.grad(trail), ufl.grad(test))) * ufl.dx
        lhs = dolfinx.fem.form(ufl.lhs(form))
        rhs = dolfinx.fem.form(ufl.rhs(form))
        boundary_facets_x0 = dolfinx.mesh.locate_entities_boundary(self.domain.mesh, self.domain.mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0))
        boundary_dofs_x0 = dolfinx.fem.locate_dofs_topological(self.vector_space.sub(0), self.domain.mesh.topology.dim - 1, boundary_facets_x0)
        bc_x0 = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs_x0, self.vector_space.sub(0))

        boundary_facets_y0 = dolfinx.mesh.locate_entities_boundary(self.domain.mesh, self.domain.mesh.topology.dim - 1, lambda x: np.isclose(x[1], 0))
        boundary_dofs_y0 = dolfinx.fem.locate_dofs_topological(self.vector_space.sub(1), self.domain.mesh.topology.dim - 1, boundary_facets_y0)
        bc_y0 = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs_y0, self.vector_space.sub(1))
        problem = dolfinx.fem.petsc.LinearProblem(
            lhs,
            rhs,
            bcs=([bc_x0, bc_y0] if self.sym_bcs else []),
            u=self.grad_ϕ,
            petsc_options={"ksp_type": "gmres", "pc_type": "hypre"},
            petsc_options_prefix="gradient_"
        )
        problem.solve()
    
    def build_normal_problem(self):
        α = 1e-2
        β = 10
        trail = ufl.TrialFunction(self.vector_space)
        test = ufl.TestFunction(self.vector_space)
        g = ufl.grad(self.ϕ)
        ϵ = self.ϵ
        m = ϵ * g / ufl.sqrt(ϵ*ϵ*ufl.inner(g, g) + α*α*ufl.exp(-β*ϵ*ϵ*ufl.inner(g, g)))

        # Build the boundary normal
        m_norm_mag = dolfinx.fem.Function(self.scalar_space)
        interpolate_expression(
            ϵ * ufl.sqrt(ufl.inner(g, g)) / ufl.sqrt(ϵ*ϵ*ufl.inner(g, g) + α*α*ufl.exp(-β*ϵ*ϵ*ufl.inner(g, g))),
            m_norm_mag
        )
        bc_fun_x0 = dolfinx.fem.Function(self.vector_space)
        bc_fun_y0 = dolfinx.fem.Function(self.vector_space)
        e1 = dolfinx.fem.Constant(self.vector_space.mesh, dolfinx.default_scalar_type((1, 0)))
        e2 = dolfinx.fem.Constant(self.vector_space.mesh, dolfinx.default_scalar_type((0, 1)))
        interpolate_expression(-m_norm_mag * e2, bc_fun_x0)
        interpolate_expression(-m_norm_mag * e1, bc_fun_y0)

        lhs = ufl.inner(trail, test) * ufl.dx
        rhs = ufl.inner(m, test) * ufl.dx
        zero_bc_val = np.array([0.0, 0.0])
        domain = self.domain.mesh
        # facets_x0 = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, 
        #                                         lambda x: np.isclose(x[0], 0))
        # facets_y0 =dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, 
        #                                         lambda x: np.isclose(x[1], 0))
        facets_x1 = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, 
                                                lambda x: np.isclose(x[0], 2))
        facets_y1 =dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim-1, 
                                                 lambda x: np.isclose(x[1], 2))
        # dofs_x0 = dolfinx.fem.locate_dofs_topological(self.n.function_space, domain.topology.dim-1, facets_x0)
        # dofs_y0 = dolfinx.fem.locate_dofs_topological(self.n.function_space, domain.topology.dim-1, facets_y0)
        dofs_x1 = dolfinx.fem.locate_dofs_topological(self.n.function_space, domain.topology.dim-1, facets_x1)
        dofs_y1 = dolfinx.fem.locate_dofs_topological(self.n.function_space, domain.topology.dim-1, facets_y1)

        boundary_facets_x0 = dolfinx.mesh.locate_entities_boundary(self.domain.mesh, self.domain.mesh.topology.dim - 1, lambda x: np.isclose(x[0], 0))
        boundary_dofs_x0 = dolfinx.fem.locate_dofs_topological(self.vector_space.sub(0), self.domain.mesh.topology.dim - 1, boundary_facets_x0)
        bc_x0 = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs_x0, self.vector_space.sub(0))

        boundary_facets_y0 = dolfinx.mesh.locate_entities_boundary(self.domain.mesh, self.domain.mesh.topology.dim - 1, lambda x: np.isclose(x[1], 0))
        boundary_dofs_y0 = dolfinx.fem.locate_dofs_topological(self.vector_space.sub(1), self.domain.mesh.topology.dim - 1, boundary_facets_y0)
        bc_y0 = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), boundary_dofs_y0, self.vector_space.sub(1))

        # bc_x0 = dolfinx.fem.dirichletbc(bc_fun_x0, dofs_x0)
        # bc_y0 = dolfinx.fem.dirichletbc(bc_fun_y0, dofs_y0)
        bc_x1 = dolfinx.fem.dirichletbc(zero_bc_val, dofs_x1, self.n.function_space)
        bc_y1 = dolfinx.fem.dirichletbc(zero_bc_val, dofs_y1, self.n.function_space)
        self.normal_problem = dolfinx.fem.petsc.LinearProblem(
            dolfinx.fem.form(lhs),
            dolfinx.fem.form(rhs),
            bcs=([bc_x0, bc_y0] if self.sym_bcs  else []),
            u=self.n,
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
            petsc_options_prefix="normal_"
        )


    def build_gls_advection_problem(self, u: dolfinx.fem.Function):
        trail, test = self.trial, self.test

        dt = self.dt
        ϕ = self.ϕ
        c = 1
        δ = c * ufl.CellDiameter(ϕ.function_space.mesh) / (2 * ufl.sqrt(ufl.inner(u, u) +  1e-10**2))
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
            petsc_options={"ksp_type": "gmres", "pc_type": "hypre"},
            petsc_options_prefix="gls_advection_"
        )


    def build_reinit_problem(self, use_mesh_eps=True):
        dτ = self.dτ
        ϕ = ufl.min_value(ufl.max_value(self.ϕ, 0.0), 1.0)
        trial, test = self.trial, self.test
        n = self.n
        ϵ = self.ϵ
        n_mesh = ufl.FacetNormal(self.domain.mesh)
        form = (
            1/dτ*(trial-ϕ)*test
            +1/2*ufl.div((trial+ϕ-2*trial*ϕ)*n)*test
            +ϵ/2*ufl.inner(ufl.grad(trial+ϕ),n)*ufl.inner(ufl.grad(test), n)
            +ϵ/2*ufl.inner((1-ufl.inner(n,n))*ufl.grad(trial+ϕ), ufl.grad(test))
        ) * ufl.dx# + 1/self.h_min * ufl.inner(ufl.grad(ϕ), n_mesh) * ufl.ds

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
        self.build_gls_advection_problem(u)
        self.build_normal_problem()
        self.build_reinit_problem(use_mesh_eps=False)

    def reinit(self, show=False):
        for i in range(self.max_reinit_iters):
            self.reinit_problem.solve()
            res = dolfinx.fem.form((self.phi_temp - self.ϕ) * (self.phi_temp - self.ϕ) * ufl.dx)
            res = dolfinx.fem.assemble_scalar(res)
            if show:
                print("LSRE tol:", res)
            self.phi_temp.x.array[:] = self.ϕ.x.array
            if res < self.reinit_tol:
                if show:
                    print(f"LS-REINIT converged after {i + 1} iterations.")
                return



    def advect(self, show=False):
        self.gls_advection_problem.solve()
        self.normal_problem.solve()
        self.phi_temp.x.array[:] = self.ϕ.x.array
        self.reinit(show=show)

       
        


        
    

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

    def remesh(self, solver: ConservativeLevelSet, interface_tol=0.1, sample_rate=1, show=False):
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
            coords = np.concatenate(coords)
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

            view_data = np.column_stack((coords, phi_values)).flatten()
            phi_view = gmsh.view.add("phiView")
            gmsh.view.add_list_data(phi_view, "SP", len(coords), view_data)

            phi_field = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(phi_field, "ViewIndex", 0)



            #coords = np.concatenate(coords)
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
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 1.5 * 1.5 * self.h_min)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 6 * 20 * self.h_min)
            

            gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

            gmsh.model.mesh.setRecombine(2, surf) 
            gmsh.option.setNumber("Mesh.Algorithm", 8) # Frontal-Delaunay for Quads
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2) # simple or blossomed
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0) # 1: all quads, 0: none

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
        ϵ = dolfinx.fem.Constant(scalar_space.mesh, 1.5 * self.h_min)
        # Update spaces and functions of solver
        solver.scalar_space, solver.vector_space, solver.ϕ, solver.ϵ, solver.κ, solver.n, solver.h, solver.grad_ϕ = scalar_space, vector_space, ϕ, ϵ, κ, n, h, grad_ϕ
        # Update the mesh and tree of the domain
        self.mesh = new_mesh
        solver.ϕ.name = "phi"
        self.tree = dolfinx.geometry.bb_tree(new_mesh, 2)

       



   


        

        

            
