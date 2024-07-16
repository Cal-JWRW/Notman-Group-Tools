import numpy as np
import numpy as np
from scipy.linalg import eig
import multiprocessing
from . import Build_Matrices
from . import DHAM
import os
from tqdm import tqdm

def Parallel_Convergence(args):
    '''
    Markov model multiprocessing framework for computing Markov model convergence

    AKA: PADHAM (Parallel-Accelerated-Dynamic-Histogram-Analysis-Method)

    Supply argument list to multiprocessing of the following format:
    args[0] = Number of bins
    args[1] = List of Trajectories
    args[2] = Bin Centers
    args[3] = Lagtime
    args[4] = min_state of colletive variable
    args[5] = max_state of collective variable
    args[6] = Temperature
    args[7] = Biasing Potential
    '''
    bins = args[0]
    Trajectories = args[1]
    Bin_Centers = args[2]
    lag = args[3]
    min_state = args[4]
    max_state = args[5]
    Temp = args[6]
    Bias = args[7]

    c_ij = []
    bias_i = []

    states = np.linspace(min_state, max_state, bins)
    for count,j in enumerate(Trajectories):
        c_ij.append(Build_Matrices.Count_Matrix(j, lag, states))
        bias_i.append(Build_Matrices.Bias_Matrix(j,Bias,states,Bin_Centers[count]))

    c_ij = np.asarray(c_ij)
    bias_i = np.asarray(bias_i)

    MM = DHAM.Construct_MM(c_ij, bias_i.T, Temp)
    d, v = eig(MM.T)
    mpeq = v[:, np.where(d == np.max(d))[0][0]]
    mpeq = mpeq / np.sum(mpeq)
    mU2 = -Temp*1.9872041E-3 * np.log(mpeq)
    mU2 -= np.min(mU2[:int(len(states))])


    return(d)

def Relaxation_Times(Trajectories,Bin_Centers,Bins, Min_State, Max_State, Biasing_Potential, Temp, Min_Lag, Max_Lag, Lag_Step):
    """
    Computes the ideal lagtime by calculating the longest relaxation times at varying lagtimes utilizing parallel acceleration

    PADHAM: Parallel-Accelerated-Dynamic-Histogram-Analysis-Method

    Trajectories: A list of umbrella sampling pullx trajectories
    Bin Centers: The umbrella centers for each simulation
    Bins: The number of bins to use
    Min_State: Minimum of the c.v 
    Max_State: Maximum of the c.v
    Biasing_potential: Biasing potential in correct units (kcal/mol/A^2)
    Temp: Simulation temperature in kelvin
    Min_Lag: Minimum lagtime to compute
    Max_Lag: Maximum lag time to compute
    Lag_Step: Step between calculations
    """
    lags = np.arange(Min_Lag, Max_Lag+1, Lag_Step)
    args = []
    MMs = []
    for i in lags:
        args.append([Bins, Trajectories[:, :], Bin_Centers, int(i), Min_State, Max_State, Temp, Biasing_Potential])

    
    with multiprocessing.Pool(processes=int(os.cpu_count()/4)) as pool:
        results = pool.map(Parallel_Convergence, args)
        MMs.extend(results)


    return(MMs)