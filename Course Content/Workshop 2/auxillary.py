# !/bin/usr/python

"""This file contains auxillary functions required for Workshop 2.
These will be imported into the main notebook so you do not need to worry about the implementation details.

Thomas Williamson, School of Engingeering
(c) 2025 The University of Edinburgh 
License: CC-BY, Creative Commons by Attribution

A previous version of this course was developed by David Ingram, School of Engineering, University of Edinburgh.
"""

import numpy as np
try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

class Grid:
    """A class defining a 2D grid on which we can implement the Jacobi and SOR iteration schemes."""

    def __init__(self, ni: int, nj: int):
        self.ni = ni
        self.nj = nj
        self.origin = (0.0, 0.0)
        self.extent = (1.0, 1.0)

        self.u = np.zeros((ni, nj))
        self.x = np.zeros((ni, nj))
        self.y = np.zeros((ni, nj))
    
    def set_origin(self, x0: float, y0: float):
        """Set the origin of the grid."""
        self.origin = (x0, y0)

    def set_extent(self, x1: float, y1: float):
        """Set the extent of the grid."""
        self.extent = (x1, y1)

    def Delta_x(self) -> float:
        """The spacing in the x-direction."""
        return (self.extent[0] - self.origin[0]) / (self.ni - 1)
    
    def Delta_y(self) -> float:
        """The spacing in the y-direction."""
        return (self.extent[1] - self.origin[1]) / (self.nj - 1)
    
    def generate(self, Quiet: bool = True):
        '''generate a uniformly spaced grid covering the domain from the
        origin to the extent.  We are going to do this using linspace from
        numpy to create lists of x and y ordinates and then the meshgrid
        function to turn these into 2D arrays of grid point ordinates.'''
        x_ord = np.linspace(self.origin[0], self.extent[0], self.ni, endpoint=True)
        y_ord = np.linspace(self.origin[1], self.extent[1], self.nj, endpoint=True)
        self.x, self.y = np.meshgrid(x_ord,y_ord)
        self.x = np.transpose(self.x)
        self.y = np.transpose(self.y)

        if not Quiet:
            print(self)
    
    def __str__(self):
        """A quick function to tell us about the grid. This will be what is displayed if you try to print the Grid object."""
        return f"Grid Object: Uniform {self.ni}x{self.nj} grid from {self.origin} to {self.extent}."


def u_0(u_E: float, u_W: float, u_N: float, u_S: float, delta_x: float = 1., delta_y: float = 1.) -> float:
    """The Jacobi iteration for the initial guess of the solution."""
    beta = delta_x / delta_y
    return (u_E + u_W + beta**2 * (u_N + u_S)) / (2 * (1 + beta**2))

def calc_Q(mesh: Grid) -> float:
    """Calculate the value of the integral quantitiy Q to determine the convergence of the scheme."""
    return simpson(simpson(np.square(mesh.u), x = mesh.y[0,:]), x = mesh.x[:,0])

def Jacobi_iteration_vs_error(mesh: Grid, iterations_to_sample: list[int] | np.ndarray[float]) -> tuple[np.ndarray[int], np.ndarray[float]]:
    if type(iterations_to_sample) == list:
        iterations_to_sample = np.array(iterations_to_sample, dtype=int)
    iterations = []
    qs = []

    n_iterations = 0

    # Now we are all set up, we can implement the iteration, similar to how we have done before.
    max_iterations = np.max(iterations_to_sample)
    u_new = mesh.u.copy()
    while n_iterations <= max_iterations:
        u_new[1:-1, 1:-1] = u_0(mesh.u[2:, 1:-1], mesh.u[:-2, 1:-1], mesh.u[1:-1, 2:], mesh.u[1:-1, :-2], mesh.Delta_x(), mesh.Delta_y())

        mesh.u = u_new.copy()
        if n_iterations in iterations_to_sample:
            iterations.append(n_iterations)
            qs.append(calc_Q(mesh))
        
        n_iterations += 1

    return np.array(iterations, dtype=int), np.array(qs, dtype=float)

def SOR_iteration_vs_error(mesh: Grid, iterations_to_sample: list[int] | np.ndarray[float]) -> tuple[np.ndarray[int], np.ndarray[float]]:
    if type(iterations_to_sample) == list:
        iterations_to_sample = np.array(iterations_to_sample, dtype=int)
    iterations = []
    qs = []

    max_iterations = np.max(iterations_to_sample)

    lamda = (np.cos(np.pi/mesh.ni)+np.cos(np.pi/mesh.nj))**2/4
    omega = 2/(1+np.sqrt(1-lamda))
    
    # calculate the coefficients
    beta = mesh.Delta_x()/mesh.Delta_y()
    beta_sq = beta**2
    C_beta = 1/(2*(1+beta_sq))
    
    # initialise u_new 
    u_new = mesh.u.copy()

    n_iterations = 0
    
    # itteration
    while n_iterations < max_iterations:
        for i in range(1,mesh.ni-1):
            for j in range(1,mesh.nj-1):
                u_new[i,j] = (1-omega)*mesh.u[i,j] + \
                        omega * C_beta*(u_new[i,j-1]+mesh.u[i,j+1]+ \
                        beta_sq*(u_new[i-1,j]+mesh.u[i+1,j]))
                
        mesh.u = u_new.copy()
        if n_iterations in iterations_to_sample:
            iterations.append(n_iterations)
            qs.append(calc_Q(mesh))
        
        n_iterations += 1

    return np.array(iterations, dtype=int), np.array(qs, dtype=float)

if __name__ == "__main__":
    print("This is an auxillary module for Workshop 2. You do not need to run this file.")