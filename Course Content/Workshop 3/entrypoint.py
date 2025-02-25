# ! /bin/usr/python

# This script provides functions for the workshops for some of the questions.
# You should try the questions yourself first but if you have not managed some of the exercieses, you can import functions from here so that you can move on.

# Script by Thomas Williamson


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

class Grid:
    """A class defining a 2D grid on which we can implement the Jacobi, SOR and matrix schemes."""

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
        self.x, self.y = np.meshgrid(x_ord, y_ord)
        self.x = np.transpose(self.x)
        self.y = np.transpose(self.y)

        if not Quiet:
            print(self)
    
    def __str__(self):
        """A quick function to tell us about the grid. This will be what is displayed if you try to print the Grid object."""
        return f"Grid Object: Uniform {self.ni}x{self.nj} grid from {self.origin} to {self.extent}."

def determine_coefficients(mesh: Grid) -> tuple[float, float]:
    """Determine the coefficients r_x and r_y to assemble the matrix for the Laplace equation.

    Parameters
    ----------

    mesh : Grid
        The mesh on which we are solving the Laplace equation.
    
    Returns
    -------

    (r_x, r_y) : tuple[float, float]
        The coefficients r_x and r_y required to assemble the matrix.
    """
    beta_squ = (mesh.Delta_x()/mesh.Delta_y())**2

    r_x = -1/(2*(1+beta_squ))
    r_y = beta_squ * r_x
    return (r_x, r_y)



def assemble_matrix(mesh: Grid):

    N = (mesh.ni - 2) * (mesh.nj - 2)

    # Create the matrix and calculate the coefficients
    # We store A as a sparse matrix to save memory
    A_mat = sp.lil_matrix((N, N), dtype=float)
    b_vec = np.zeros(N, dtype = float)

    R_x, R_y = determine_coefficients(mesh)

    # Assemble the matrix A and the vector b

    for j in range(1, mesh.nj-1):
        for i in range(1, mesh.ni-1):
            
            # calculate the k index
            k = (i-1) + (mesh.ni-2)*(j-1)
            
            # leading diagonal coeficient
            A_mat[k,k]=1
                
            # East boundary
            if i<mesh.ni-2:
                A_mat[k,k+1]=R_x
            else:
                b_vec[k] += -R_y*mesh.u[i+1,j]
                
            # West boundary
            if i>1:
                A_mat[k,k-1]=R_x
            else:
                b_vec[k] += -R_y*mesh.u[i-1,j]

            # North boundary
            if j<mesh.nj-2:
                A_mat[k,k+(mesh.ni-2)]=R_y
            else:
                b_vec[k] += -R_y*mesh.u[i,j+1]

            # South boundary
            if j>1:
                A_mat[k,k-(mesh.ni-2)]=R_y
            else:
                b_vec[k] += -R_y*mesh.u[i,j-1]


    return A_mat, b_vec

if __name__ == '__main__':
    print("This script provides functions for the workshops for some of the questions. You do not need to run it directly.\nPlease see the Jupyter notebook for the workshop exercises.")