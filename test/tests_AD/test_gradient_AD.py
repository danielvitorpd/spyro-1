from spyro.pml import damping
import numpy as np
from pyadjoint import enlisting
import matplotlib.pyplot as plt
from firedrake import *

import spyro
from spyro.domains import quadrature
from firedrake_adjoint import *
#from inputfiles.Model1_gradient_2d import model
#from inputfiles.Model1_gradient_2d_pml import model_pml
from scipy.ndimage import gaussian_filter

import copy
# outfile_total_gradient = File(os.getcwd() + "/results/Gradient.pvd")
OMP_NUM_THREADS=1
forward = spyro.solvers.forward
gradient = spyro.solvers.gradient
functional = spyro.utils.compute_functional

model = {}

model["opts"] = {
    "method": "KMV",  # either CG or KMV
    "quadratrue": "KMV",  # Equi or KMV
    "degree": 2,  # p order
    "dimension": 2,  # dimension
}
model["parallelism"] = {
    "type": "spatial",
    "custom_cores_per_shot": [],
    "num_cores_per_shot": 1
}
model["mesh"] = {
    "Lz": 1.,  # depth in km - always positive
    "Lx": 1.,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "square.msh",
    "initmodel": "not_used.hdf5",
    "truemodel": "velocity_models/marmousi_velocity_model.hdf5",
}
model["BCs"] = {
    "status": False,  # True or false
    "outer_bc": "non-reflective",  #  None or non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,  # damping layer has a exponent variation
    "cmax": 4.5,  # maximum acoustic wave velocity in PML - km/s
    "R": 1e-6,  # theoretical reflection coefficient
    "lz": 0.9,  # thickness of the PML in the z-direction (km) - always positive
    "lx": 0.9,  # thickness of the PML in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the PML in the y-direction (km) - always positive
    "method": "none",
}
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": 1,
    "source_pos": spyro.create_transect((-0.01, 1.0), (-0.01, 15.0), 1),
    "frequency": 5.0,
    "delay": 1.0,
    "num_receivers": 500,
    "receiver_locations": spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), 500),
}
model["timeaxis"] = {
    "t0": 0.0,  #  Initial time for event
    "tf": 1.00,  # Final time for event
    "dt": 0.0005,
    "amplitude": 1,  # the Ricker has an amplitude of 1.
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 99999,  # how frequently to save solution to RAM
}

def _make_vp_exact(V, mesh):
    """Create a circle with higher velocity in the center"""
    z, x = SpatialCoordinate(mesh)
    #vp_exact = spyro.io.interpolate(model, mesh, V, guess=False)
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_exact_damping(V, mesh):
    """Create a half space"""
    z, x = SpatialCoordinate(mesh)
    #velocity = conditional(z > -0.5, 2, 4)
    velocity = x**2 + z**2 + 4.0
    vp_exact = Function(V, name="vp").interpolate(velocity)
    File("exact_vel.pvd").write(vp_exact)
    return vp_exact


def _make_vp_guess(V, mesh):
    """The guess is a uniform velocity of 4.0 km/s"""
    z, x = SpatialCoordinate(mesh)
    v0 = Function(V).interpolate(4.0 + 0.0 * x)

    sigma = 10

    v0.dat.data[:] = gaussian_filter(v0.dat.data[:], sigma = sigma)

    File("guess_vel.pvd").write(v0)
    
    return v0

def test_gradient():
    _test_gradient(model)

def test_gradient_damping():
    _test_gradient(model, damping=True)


def _test_gradient(options, damping=False):
    with stop_annotating():
        comm = spyro.utils.mpi_init(options)

        mesh, V = spyro.io.read_mesh(options, comm)
        num_rec = options["acquisition"]["num_receivers"]
        if damping:
            vp_exact = _make_vp_exact_damping(V, mesh)
            δs = np.linspace(0.1, 0.9, num_rec)
            X, Y = np.meshgrid(-0.1, δs)
        else:
            #vp_exact = _make_vp_exact(V, mesh)
            #vp_exact = spyro.io.interpolate(options, mesh, V, guess=False)
            #vp_exact = Function(V)
            #File("exact_vel.pvd").write(vp_exact)
            
            # create_transect((0.1, -2.90), (2.9, -2.90), 100)
            δs = np.linspace(0.1, 2.9, num_rec)
            X, Y = np.meshgrid(δs,-2.90)

        xs = np.vstack((X.flatten(), Y.flatten())).T
           
        sources = spyro.Sources(options, mesh, V, comm)  
        solver  = spyro.solver_AD()
    
        # simulate the exact options
        solver.p_true_rec = solver.forward_AD(options, mesh, comm,
                                vp_exact, sources, xs)

    
    vp_guess = _make_vp_guess(V, mesh)
    control = Control(vp_guess)
    solver.Calc_Jfunctional = True
    p_rec_guess = solver.forward_AD(options, mesh, comm,
                               vp_guess, sources, xs)
     
    J  = solver.obj_func
    
    dJ   = compute_gradient(J, control)
    Jhat = ReducedFunctional(J, control) 

    with stop_annotating():
        Jm = copy.deepcopy(J)

        qr_x, _, _ = quadrature.quadrature_rules(V)

        print("\n Cost functional at fixed point : " + str(Jm) + " \n ")
        
        with open('gradients.txt', 'a') as f:

            f.write("\n Cost functional at fixed point : " + str(Jm) + " \n ")

        File("gradient.pvd").write(dJ)

        steps = [1e-3, 1e-4, 1e-5]  # , 1e-6]  # step length
        

        delta_m = Function(V)  # model direction (random)
        delta_m.assign(dJ)
        derivative = enlisting.Enlist(Jhat.derivative())
        hs = enlisting.Enlist(delta_m)

        projnorm = sum(hi._ad_dot(di) for hi, di in zip(hs, derivative))
        
        # this deepcopy is important otherwise pertubations accumulate
        vp_original = vp_guess.copy(deepcopy=True)

        errors = []
        for step in steps:  # range(3):
            
            solver.obj_func   = 0.
            # J(m + delta_m*h)
            vp_guess = vp_original + step*delta_m
            p_rec_guess = solver.forward_AD(options, mesh, comm,
                                vp_guess, sources, xs)  
            Jp = solver.obj_func
            fd_grad = (Jp - Jm) / step
            print(
                "\n Cost functional for step "
                + str(step)
                + " : "
                + str(Jp)
                + ", fd approx.: "
                + str(fd_grad)
                + ", grad'*dir : "
                + str(projnorm)
                + " \n ",
            )

            with open('gradients.txt', 'a') as f:

                f.write("\n Cost functional for step "
                + str(step)
                + " : "
                + str(Jp)
                + ", fd approx.: "
                + str(fd_grad)
                + ", grad'*dir : "
                + str(projnorm)
                + " \n ")
            
            errors.append(100 * ((fd_grad - projnorm) / projnorm))
            # step /= 2

        # all errors less than 1 %
        errors = np.array(errors)
        assert (np.abs(errors) < 5.0).all()

if __name__ == "__main__":

    receivers = [100, 200, 300, 400, 500]
    
    for receiver in receivers:
        
        model["acquisition"]["num_receivers"] = receiver
        model["acquisition"]["receiver_locations"] = spyro.create_transect((-0.10, 0.1), (-0.10, 17.0), receiver)
        test_gradient_damping()
    #or test_gradient_damping() #when the damping is employed

