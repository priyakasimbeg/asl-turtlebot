import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    path = np.array(path)
    x = path[:, 0]
    y = path[:, 1]
    dx = np.array([xb - xa for xb, xa in zip(x[1:], x[:-1])])
    dy = np.array([yb - ya for yb, ya in zip(y[1:], y[:-1])])
    dt_s = np.sqrt(dx**2 + dy**2)/V_des
    t = np.cumsum(dt_s)
    t = np.hstack(([0], t))

    splx = scipy.interpolate.splrep(t, x, k=3, s=alpha)
    sply = scipy.interpolate.splrep(t, y, k=3, s=alpha)

    t_smoothed = np.arange(0, t[-1]+dt, dt)

    x = scipy.interpolate.splev(t_smoothed, splx, )
    y = scipy.interpolate.splev(t_smoothed, sply, )

    x_dot = scipy.interpolate.splev(t_smoothed, splx, der=1 )
    y_dot = scipy.interpolate.splev(t_smoothed, sply, der=1)

    x_ddot = scipy.interpolate.splev(t_smoothed, splx, der=2)
    y_ddot = scipy.interpolate.splev(t_smoothed, sply, der=2)

    theta = np.arctan(y_dot/x_dot)
    traj_smoothed = np.column_stack((x, y, theta, x_dot, y_dot, x_ddot, y_ddot))

    ########## Code ends here ##########

    return traj_smoothed, t_smoothed
