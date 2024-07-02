import numpy as np
import six
from collections import Counter

def Count_Matrix(traj, lag, states):

    """
    determine transition counts from a trajectory with a given lag time

    C[i,j] where i is the intial state and j the product state.

    The first row contains thus all the transitions from state 0.
    The first colmun C[:,0] all transition into state 0.

    """
    n_states = len(states)
    traj = np.digitize(traj, states)-1
    if n_states is None:
        n_states = np.max(traj)
    b = np.zeros((n_states, n_states))

    for (i,j), c in six.iteritems(Counter(zip(traj[:-lag], traj[lag:]))):
        b[int(i), int(j)] = c

    return(b)

def Bias_Matrix(traj, bias_potential, states, cen):
    umbrella_center = cen
    bias = np.zeros(len(states))
    qp = (states[1]-states[0])/2
    for counter, i in enumerate(states):
        bias[counter] = bias_potential*((umbrella_center-i-qp)**2)*0.5*(1/(303.15*1.9872041E-3))


    return(bias)

def count_transitions(b, numbins, lagtime, endpt=None):
    if endpt is None:
        endpt = b
    Ntr = np.zeros(shape=(b.shape[0], numbins, numbins), dtype=int)  # number of transitions
    for k in range(b.shape[0]):
        for i in range(lagtime, b.shape[1]):
            try:
                Ntr[k, b[k, i - lagtime] - 1, endpt[k, i] - 1] += 1
            except IndexError:
                continue
    sumtr = np.sum(Ntr, axis=0)
    trvec = np.sum(Ntr, axis=2)
    # sym = 0.5 * (sumtr + np.transpose(sumtr))
    # anti = 0.5 * (sumtr - np.transpose(sumtr))
    # print("Degree of symmetry:",
    #       (np.linalg.norm(sym) - np.linalg.norm(anti)) / (np.linalg.norm(sym) + np.linalg.norm(anti)))
    return sumtr, trvec
