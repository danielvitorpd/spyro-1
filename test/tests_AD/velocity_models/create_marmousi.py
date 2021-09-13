from mpi4py import MPI
import meshio

from SeismicMesh import get_sizing_function_from_segy, generate_mesh, Rectangle, write_velocity_model
fname = "Mar2_Vp_1.25m.segy"

bbox = (-3500.0, 0.0, 0.0, 17000.0)  # o tamanho do domínio

write_velocity_model(

    fname,

    ofname="gato_velocity_model",  # como o arquivo será chamado

    bbox=bbox,

    domain_pad=500,  # a largura da camada PML em metros

    pad_style="edge",  # a forma como a velocidade é estendida para a camada

    units="km-s",  # as unidades do modelo de velocidade

)
comm = MPI.COMM_WORLD

# Desired minimum mesh size in domain
hmin = 40.0

rectangle = Rectangle(bbox)

# Construct mesh sizing object from velocity model
ef = get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    wl=10,
    freq=7,
    dt=0.001,
    grade=0.15,
    domain_pad=1e3,
    pad_style="edge",
)

points, cells = generate_mesh(domain=rectangle, edge_length=ef)

if comm.rank == 0:
    meshio.write_points_cells(
        "marmousi_mesh.msh",
        points / 1000,
        [("triangle", cells)],
        file_format="gmsh22",
        binary=False
    )
