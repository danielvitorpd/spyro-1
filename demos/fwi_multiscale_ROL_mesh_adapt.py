from firedrake import *

import SeismicMesh
import meshio

import numpy as np
from ROL.firedrake_vector import FiredrakeVector as FeVector
import ROL
import spyro
from mpi4py import MPI


outdir = "testing_fwi/"

# START DEFINING OPTIONS
model = {}
# Specify Finite Element related options
model["opts"] = {
    "method": "KMV",  # CG, DG, KMV
    "variant": None,
    "type": "SIP",  # for DG only - SIP, NIP and IIP
    "element": "tria",  # tria or tetra
    "degree": 3,  # spatial polynomial order
    "quadrature": "KMV",  # # GLL, GL, Equi, KMV
    "dimension": 2,  # spatial dimension
    "beta": 0.0,  # for Newmark time integration only
    "gamma": 0.5,  # for Newmark time integration only
}
# Define the mesh geometry and filenames of the velocity models
model["mesh"] = {
    "Lz": 4.0,  # depth in km - always positive
    "Lx": 18.0,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/mm_init.msh",
    "initmodel": "velocity_models/mm_guess.hdf5",
    "truemodel": "velocity_models/mm_exact.hdf5",
}
# Use a Perfectly Matched Layer to damp reflected waves.
# Note here, it's built to be 0.5 km thick on three sides of the domain
model["PML"] = {
    "status": True,  # True or false
    "outer_bc": "non-reflective",  #  neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial. hyperbolic, shifted_hyperbolic
    "exponent": 1,
    "cmax": 4.7,  # maximum acoustic wave velocity in PML - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.5,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.5,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}
# Simulate with 40 shots equi-spaced near the top of the domain
# and record the solution at 301 receivers.
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 20,
    "source_pos": spyro.create_receiver_transect((-0.15, 0.1), (-0.15, 16.9), 20),
    "frequency": 10.0,
    "delay": 1.0,
    "num_receivers": 301,
    "receiver_locations": spyro.create_receiver_transect(
        (-0.15, 0.1), (-0.15, 16.9), 301
    ),
}
# Perform each shot simulation for 3.0 seconds and save all
# timesteps for the gradient calculation.
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 3.0,  # Final time for event
    "dt": 0.0005,  # timestep size
    "nspool": 200,  # how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to RAM
}
# Use one core per shot.
model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}

model["inversion"] = {"freq_bands": [2.0]}


def remesh(fname, freq, mesh_iter):
    """for now some hardcoded options"""
    bbox = (-4000.0, 0.0, -500.0, 17500.0)

    wl = 10

    # Desired minimum mesh size in domain
    hmin = 1500.0 / (wl * freq)

    rectangle = SeismicMesh.Rectangle(bbox)

    # Construct mesh sizing object from velocity model
    ef = SeismicMesh.get_sizing_function_from_segy(
        fname,
        bbox,
        hmin=hmin,
        wl=wl,
        freq=freq,
        dt=0.001,
    )

    SeismicMesh.write_velocity_model(
        fname, ofname="velocity_models/mm_GUESS" + str(mesh_iter) + ".hdf5"
    )

    points, cells = SeismicMesh.generate_mesh(domain=rectangle, edge_length=ef)

    meshio.write_points_cells(
        "meshes/mm_GUESS" + str(mesh_iter) + ".msh",
        points / 1000,
        [("triangle", cells)],
        file_format="vtk",
    )


comm = spyro.utils.mpi_init(model)

mesh_iter = 0

for index, freq_band in enumerate(model["inversion"]["freq_bands"]):
    if comm.comm.rank == 0 and comm.ensemble_comm.rank == 0:
        print(
            "INFO: Executing inversion for low-passed cut off of "
            + str(freq_band)
            + " Hz...",
            flush=True,
        )

    # do the mesh adaptation here based on the new guess mesh
    if mesh_iter > 0:

        # interpolate vp_exact to a structured grid and write to a segy file for later meshing with SeismicMesh
        xi, yi, vp_i = spyro.utils.write_function_to_grid(
            vp_guess, V, grid_spacing=1.0 / 1000.0
        )

        segy_fname = "velocity_models/mm_GUESS" + str(mesh_iter) + ".segy"

        spyro.io.create_segy(vp_guess, segy_fname)

        # call SeismicMesh in serial to build a new mesh of the domain based on new_segy
        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            remesh(segy_fname, freq_band, mesh_iter)

        # point to latest mesh file
        model["mesh"]["meshfile"] = "meshes/mm_GUESS" + str(mesh_iter) + ".msh"

        # point to the latest guess velocity model
        model["mesh"]["initmodel"] = (
            "velocity_models/mm_GUESS" + str(mesh_iter) + ".hdf5"
        )

    # Given the new mesh, we need to reinitialize some things
    mesh, V = spyro.io.read_mesh(model, comm)

    vp_guess = spyro.io.interpolate(model, mesh, V, guess=True)

    sources = spyro.Sources(model, mesh, V, comm).create()

    receivers = spyro.Receivers(model, mesh, V, comm).create()

    water = np.where(vp_guess.dat.data[:] < 1.51)

    qr_x, _, _ = spyro.domains.quadrature.quadrature_rules(V)

    if comm.ensemble_comm.rank == 0:
        control_file = File(
            outdir + "control" + str(freq_band) + "Hz+.pvd", comm=comm.comm
        )
        grad_file = File(outdir + "grad" + str(freq_band) + "Hz.pvd", comm=comm.comm)

    class L2Inner(object):
        """How ROL computes the L2 norm"""

        def __init__(self):
            self.A = assemble(
                TrialFunction(V) * TestFunction(V) * dx(rule=qr_x), mat_type="matfree"
            )
            self.Ap = as_backend_type(self.A).mat()

        def eval(self, _u, _v):
            upet = as_backend_type(_u).vec()
            vpet = as_backend_type(_v).vec()
            A_u = self.Ap.createVecLeft()
            self.Ap.mult(upet, A_u)
            return vpet.dot(A_u)

    class Objective(ROL.Objective):
        """Subclass of ROL.Objective to define value and gradient for problem"""

        def __init__(self, inner_product):
            ROL.Objective.__init__(self)
            self.inner_product = inner_product
            self.vp_guess = vp_guess
            self.p_guess = None
            self.misfit = 0.0
            self.J_local = np.zeros((1))
            self.J_total = np.zeros((1))

        def value(self, x, tol):
            """Compute the functional"""
            for sn in range(model["acquisition"]["num_sources"]):
                if spyro.io.is_owner(comm, sn):
                    # run a simulation low-pass filtering the source
                    self.p_guess, p_guess_recv = spyro.solvers.Leapfrog(
                        model,
                        mesh,
                        comm,
                        self.vp_guess,
                        sources,
                        receivers,
                        source_num=sn,
                        lp_freq_index=index,
                    )
                    p_exact_recv = spyro.io.load_shots(
                        "shots/mm_exact_" + str(10.0) + "_Hz_source_" + str(sn) + ".dat"
                    )
                    # low-pass filter the shot record for the current frequency band.
                    p_exact_recv = spyro.utils.butter_lowpass_filter(
                        p_exact_recv, freq_band, 1.0 / model["timeaxis"]["dt"]
                    )
                    # Calculate the misfit.
                    self.misfit = spyro.utils.evaluate_misfit(
                        model, comm, p_guess_recv, p_exact_recv
                    )
            J = spyro.utils.compute_functional(model, comm, self.misfit)
            self.J_local[0] = J
            # reduce over all cores
            COMM_WORLD.Allreduce(self.J_local, self.J_total, op=MPI.SUM)
            # divide by the size of the ensemble
            self.J_total[0] /= comm.ensemble_comm.size
            return self.J_total[0]

        def gradient(self, g, x, tol):
            """Compute the gradient of the functional"""
            # Check if the program has converged (and exit if so).
            # reset the functional and gradient to zero
            self.J_local[0] = np.zeros((1))
            self.dJ_local = Function(V, name="grad_local")
            self.dJ_total = Function(V, name="grad_total")
            # solve the forward problem
            for sn in range(model["acquisition"]["num_sources"]):
                if spyro.io.is_owner(comm, sn):
                    p_exact_recv = spyro.io.load_shots(
                        "shots/mm_exact_" + str(10.0) + "_Hz_source_" + str(sn) + ".dat"
                    )
                    # low-pass filter the shot record for the current frequency band.
                    p_exact_recv = spyro.utils.butter_lowpass_filter(
                        p_exact_recv, freq_band, 1.0 / model["timeaxis"]["dt"]
                    )
                    # Calculate the gradient of the functional.
                    dJ = spyro.solvers.Leapfrog_adjoint(
                        model,
                        mesh,
                        comm,
                        self.vp_guess,
                        self.p_guess,
                        self.misfit,
                        source_num=sn,
                    )
                    self.dJ_local.dat.data[:] += dJ.dat.data[:]

            # sum over all ensemble members
            comm.ensemble_comm.Allreduce(
                self.dJ_local.dat.data[:], self.dJ_total.dat.data[:], op=MPI.SUM
            )

            # mask the water layer
            self.dJ_total.dat.data[water] = 0.0

            if comm.ensemble_comm.rank == 0:
                grad_file.write(self.dJ_total)
            g.scale(0)
            g.vec += self.dJ_total
            # switch order of misfit calculation to switch this
            g.vec *= -1

        def update(self, x, flag, iteration):
            """Update the control"""
            u = Function(V, x.vec, name="velocity")
            self.vp_guess.assign(u)
            if iteration >= 0:
                if comm.ensemble_comm.rank == 0:
                    control_file.write(self.vp_guess)

    paramsDict = {
        "General": {"Secant": {"Type": "Limited-Memory BFGS", "Maximum Storage": 10}},
        "Step": {
            "Type": "Augmented Lagrangian",
            "Augmented Lagrangian": {
                "Subproblem Step Type": "Line Search",
                "Subproblem Iteration Limit": 10.0,
            },
            "Line Search": {"Descent Method": {"Type": "Quasi-Newton Step"}},
        },
        "Status Test": {
            "Gradient Tolerance": 1e-16,
            "Iteration Limit": 25,
            "Step Tolerance": 1.0e-16,
        },
    }

    params = ROL.ParameterList(paramsDict, "Parameters")

    inner_product = L2Inner()

    obj = Objective(inner_product)

    u = Function(V, name="velocity").assign(vp_guess)
    opt = FeVector(u.vector(), inner_product)

    xlo = Function(V)
    xlo.interpolate(Constant(1.0))
    x_lo = FeVector(xlo.vector(), inner_product)

    xup = Function(V)
    xup.interpolate(Constant(5.0))
    x_up = FeVector(xup.vector(), inner_product)

    bnd = ROL.Bounds(x_lo, x_up, 1.0)

    algo = ROL.Algorithm("Line Search", params)

    # this calls a sequence of processes
    algo.run(opt, obj, bnd)

    if comm.ensemble_comm.rank == 0:
        File("res" + str(freq_band) + ".pvd", comm=comm.comm).write(obj.vp_guess)

    # important: update the control for the next frequency band to start!
    vp_guess = Function(V, opt.vec)

    # increment the mesh counter
    mesh_iter += 1
