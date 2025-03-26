"""Implementation of the solver schemes of the 1D advection equation.
These are to be used in Section 3 of Workshop 5 of the PDEs3 course.

This module contains implementations of the Lax-Friedrichs, Lax-Wendroff and MacCormack schemes.

Module written by Thomas Williamson
"""


import numpy as np

def Lax_Friedrichs(us: np.ndarray, a: float, dt: float, dx: float) -> np.ndarray:

    """Implementation of the Lax-Friedrichs scheme for the 1D advection equation.
    
    Parameters
    ----------
    
    us : np.ndarray
        The current state of the system.
    a : float
        The advection speed.
    dt : float
        The time step size.
    dx : float
        The grid spacing.
    
    Returns
    -------
    
    np.ndarray
        The updated state of the system.
    """
    

    if a > 0:
        us_new = 0.5 * (np.roll(us,1) + np.roll(us,-1)) + 0.5 * a * dt / dx * (np.roll(us,1) - np.roll(us,-1))

    elif a < 0:
        us_new = 0.5 * (np.roll(us,1) + np.roll(us,-1)) - 0.5 * a * dt / dx * (np.roll(us,1) - np.roll(us,-1))

    else:
        us_new = us
    

    return us_new

def Lax_Wendroff(us: np.ndarray, a: float, dt: float, dx: float) -> np.ndarray:

    """Implementation of the Lax-Wendroff scheme for the 1D advection equation.
    
    Parameters
    ----------
    
    us : np.ndarray
        The current state of the system.
    a : float
        The advection speed.
    dt : float
        The time step size.
    dx : float
        The grid spacing.
    
    Returns
    -------
    
    np.ndarray
        The updated state of the system.
    """
    if a > 0:
        us_new = us - 0.5 * a * dt / dx * (np.roll(us,1) - np.roll(us,-1)) + 0.5 * (a * dt / dx)**2 * (np.roll(us,1) - 2*us + np.roll(us,-1))
    elif a < 0:
        us_new = us - 0.5 * a * dt / dx * (np.roll(us,1) - np.roll(us,-1)) + 0.5 * (a * dt / dx)**2 * (np.roll(us,1) - 2*us + np.roll(us,-1))
    else:
        us_new = us
    
    return us_new

def FOU(us: np.ndarray, a: float, dt: float, dx: float) -> np.ndarray:
    """Implementation of the first order upwind scheme for the 1D advection equation.
    
    Parameters
    ----------
    
    us: np.ndarray
        The current state of the system.
    a: float
        The advection speed.
    dt: float
        The time step size.
    dx: float
        The grid spacing.
        
    Returns
    -------
    
    np.ndarray
        The updated state of the system.
    """

    us_new = us - a*dt/dx*(np.roll(us, -1) - us)

    return us_new


def MacCormack(us: np.ndarray, a: float, dt: float, dx: float) -> np.ndarray:

    """Implementation of the MacCormack predictor-corrector scheme for the 1D advection equation.
    
    Parameters
    ----------
    
    us : np.ndarray
        The current state of the system.
    a : float
        The advection speed.
    dt : float
        The time step size.
    dx : float
        The grid spacing.
    
    Returns
    -------
    
    np.ndarray
        The updated state of the system.
    """
    if a > 0:

        us_predictor = us - a * dt / dx * (np.roll(us,-1) - us)

        us_new = 0.5 * (us + us_predictor) - 0.5 * a * dt / dx *(us_predictor - np.roll(us_predictor, 1))
    
    elif a < 0:
        
        us_predictor = us - a * dt / dx * (us - np.roll(us,1))

        us_new = 0.5 * (us + us_predictor) - 0.5 * a * dt / dx * (np.roll(us_predictor, -1) - us_predictor)

    return us_new

if __name__ == "__main__":
    print("This is the schemes module, it is not meant to be ran as a script. You will use these schemes as part of Section 3 in Workshop 5.")