import numpy as np
from scipy.interpolate import CubicSpline
from transforms3d.axangles import axangle2mat



# 1. Jittering
def da_jitter(X, sigma=0.05):
    """
    Do jittering for data.

    Parameters
    ------------
    sigma :
         Standard devitation (STD) of the noise.
    """
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise


# 2. Scaling
def da_scaling(X, sigma=0.1):
    """
    Scale data with a factor.

    Parameters
    ------------
    sigma :
         STD of the zoom-in/out factor.
    """
    scalingFactor = np.random.normal(
        loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


# "Scaling" can be considered as "applying constant noise to the entire samples".
# "Jittering" can be considered as "applying different noise to each sample".
# "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples".

# This example using cubic splice is not the best approach to generate random curves.
# You can use other approaches, e.g., Gaussian process regression, Bezier curve, etc.
# Random curves around 1.0.


# 3. Magnitude Warping
def generate_random_curves(X, sigma=0.2, knot=4):
    """
    Use cubic spline interpolation to generate a curve.

    Parameters
    -------------
    sigma :
        STD of the random knots for generating curves.
    knot :
        Knots for the random curves (complexity of the curves).
    """
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0,
                                                X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


def da_magwarp(X, sigma):
    return X * generate_random_curves(X, sigma)


# 4. Time Warping
def distort_time_steps(X, sigma=0.2):
    tt = generate_random_curves(X, sigma)
    tt_cum = np.cumsum(tt, axis=0)
    t_scale = [(X.shape[0]-1) / tt_cum[-1, 0], (X.shape[0]-1) /
               tt_cum[-1, 1], (X.shape[0]-1) / tt_cum[-1, 2]]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum


def da_timewarp(X, sigma=0.2):
    tt_new = distort_time_steps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
    return X_new


# 5. Rotation
def da_rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axangle2mat(axis, angle))


# 6. Permutation
def da_permutation(X, nPerm=4, minSegLength=10):
    """
    Parameters
    -------------
    nPerm :
        # of segments to permute.
    minSegLength :
        Allowable minimum length for each segment.
    """
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True

    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength,
                                               X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False

    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]] : segs[idx[ii] + 1], :]
        X_new[pp : pp+len(x_temp), :] = x_temp
        pp += len(x_temp)

    return X_new


# This approach is similar to Time Warping, but only uses subsamples (not all samples) for interpolation.
# Using TimeWarp is more recommended.

# 7. Random Sampling
def rand_sample_timesteps(X, nSample):
    """
    Parameters
    ---------------
    nSample :
        # of subsamples (nSample <= X.shape[0]).
    """
    tt = np.zeros((nSample, X.shape[1]), dtype=int)
    tt[1:-1, 0] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
    tt[1:-1, 1] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
    tt[1:-1, 2] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
    tt[-1, :] = X.shape[0] - 1
    return tt


def da_randsampling(X, nSample_rate):
    nSample = int(len(X) * nSample_rate)
    tt = rand_sample_timesteps(X, nSample)
    X_new = np.zeros(X.shape)
    X_new[:, 0] = np.interp(np.arange(X.shape[0]), tt[:, 0], X[tt[:, 0], 0])
    X_new[:, 1] = np.interp(np.arange(X.shape[0]), tt[:, 1], X[tt[:, 1], 1])
    X_new[:, 2] = np.interp(np.arange(X.shape[0]), tt[:, 2], X[tt[:, 2], 2])
    return X_new

