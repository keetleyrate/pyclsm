from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as petsc

def update_solver(solver, a, L, b, bcs, sol):
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L)
    petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    solver.solve(b, sol.vector)
    sol.x.scatter_forward()

def create_solver(mesh, lhs, rhs, bcs):
    A = petsc.assemble_matrix(lhs, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(rhs)
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType("boomeramg")
    return solver, b