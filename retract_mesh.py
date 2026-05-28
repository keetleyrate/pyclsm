import gmsh
import sys
import dolfinx
from mpi4py import MPI

def generate_contact_line_mesh(l, L, n_inner, n_outer):
    gmsh.initialize()
    gmsh.model.add("structured_quad")
    
    bl = gmsh.model.geo.add_point(0, 0, 0)
    br = gmsh.model.geo.add_point(L, 0, 0)
    tr = gmsh.model.geo.add_point(L, L, 0)
    tl = gmsh.model.geo.add_point(0, L, 0)
    mid = gmsh.model.geo.add_point(l, l, 0)

    bm = gmsh.model.geo.add_point(l, 0, 0)
    lm = gmsh.model.geo.add_point(0, l, 0)

    bl_lm = gmsh.model.geo.add_line(bl, lm)
    lm_mid = gmsh.model.geo.add_line(lm, mid)
    mid_bm = gmsh.model.geo.add_line(mid, bm)
    bm_bl = gmsh.model.geo.add_line(bm, bl)
    lm_tl = gmsh.model.geo.add_line(lm, tl)
    br_bm = gmsh.model.geo.add_line(br, bm)
    tl_tr = gmsh.model.geo.add_line(tl, tr)
    tr_br = gmsh.model.geo.add_line(tr, br)

    inner_square = gmsh.model.geo.add_curve_loop([bl_lm, lm_mid, mid_bm, bm_bl])
    inner_square_surf = gmsh.model.geo.add_plane_surface([inner_square])

    outer_loop = gmsh.model.geo.add_curve_loop([lm_tl, tl_tr, tr_br, br_bm, -mid_bm, -lm_mid])
    outer_surf = gmsh.model.geo.add_plane_surface([outer_loop])

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [outer_surf, inner_square_surf], name="")

    gmsh.model.mesh.set_transfinite_curve(bl_lm, n_inner)
    gmsh.model.mesh.set_transfinite_curve(lm_mid, n_inner)  
    gmsh.model.mesh.set_transfinite_curve(mid_bm, n_inner)
    gmsh.model.mesh.set_transfinite_curve(bm_bl, n_inner)
    gmsh.model.mesh.set_transfinite_curve(tl_tr, n_outer)
    gmsh.model.mesh.set_transfinite_curve(tr_br, n_outer)
    gmsh.model.mesh.set_transfinite_curve(lm_tl, n_outer)
    gmsh.model.mesh.set_transfinite_curve(br_bm, n_outer)
    gmsh.model.mesh.set_transfinite_surface(inner_square_surf)

    gmsh.model.mesh.set_recombine(2, inner_square_surf)
    gmsh.model.mesh.set_recombine(2, outer_surf)
    gmsh.option.setNumber("Mesh.Algorithm", 8) # Frontal-Delaunay for Quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1) # simple or blossomed
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    
  
    gmsh.model.mesh.generate(2)

    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    new_mesh_data = dolfinx.io.gmsh.model_to_mesh(
            gmsh.model, 
            MPI.COMM_WORLD, 
            rank=0, 
            gdim=2
        )

    gmsh.finalize()
    return new_mesh_data

