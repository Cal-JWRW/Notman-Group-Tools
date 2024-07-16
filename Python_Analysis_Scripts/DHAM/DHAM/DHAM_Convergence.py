import numpy as np
from scipy.linalg import eig
import multiprocessing
from . import Build_Matrices
from . import DHAM_Diffusion
from . import DHAM
import os

import warnings

# Filter out the complex warning
warnings.filterwarnings('ignore', category=Warning)

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
    d=np.asarray(d, dtype=np.float128)
    v = np.asarray(v, dtype=np.float128)
    mpeq = v[:, np.where(d == np.max(d))[0][0]]
    mpeq = mpeq / np.sum(mpeq)
    mU2 = -Temp*1.9872041E-3 * np.log(mpeq)
    mU2 -= np.min(mU2[:int(len(states))])

    return(mU2)


def Bin_Convergence(Trajectories,Bin_Centers, lag, Max_Bins, Bin_Step, min_state, max_state, Biasing_Potential, Temp, min_bins):
    """
    Calculates the markov model convergence in terms of discretization by leveraging multiprocessing

    PADHAM: Parallel-Accelerated-Dynamic-Histogram-Analysis-Method

    Trajectories: A list of umbrella sampling pullx trajectories
    Bin Centers: The umbrella centers for each simulation
    Lag: Lag time at which to construct free energy files
    Max_Bins: Maximum number of bins
    Bin_Step: Number of bins between samples
    min_state: Minimum value of the collective variable
    max_state: Maximum valuemof the collective variable
    Biasing_Potential: Biasing potential in correct units (kCal/mol/A^2)
    Temp: Simulation Temp in Kelvin 
    min_bins: Minimum number of bins to divide collective variable (Typically select the  number of windows used)
    """
    bins = np.arange(min_bins, Max_Bins, Bin_Step)
    args = []
    pmf_results = []
    for i in bins:
        args.append([i, Trajectories, Bin_Centers, lag, min_state, max_state, Temp, Biasing_Potential])
    
    with multiprocessing.Pool(processes=int(os.cpu_count()/4)) as pool:
        results = pool.map(Parallel_Convergence, args)
        pmf_results.extend(results)

    AUC = []
    for count, i in enumerate(pmf_results):
        states = np.linspace(min_state, max_state, bins[count])
        AUC.append(np.trapz(i,states))

    

    return(bins, pmf_results, AUC)

def Time_Equilibration(Trajectories, Bin_Centers, lag, Bins, Time_Step,Min_State, Max_State, Biasing_Potential, Temp, sim_len):
    """
    Calculates the free energy Equilibration in terms of including more data 

    PADHAM: Parallel-Accelerated-Dynamic-Histogram-Analysis-Method

    Trajectories: A list of umbrella sampling pullx trajectories
    Bin Centers: The umbrella centers for each simulation
    Lag: Lag time at which to construct the free energy profiles
    Bins: The number of bins to use
    Time_Step: The time step of including more data
    Min_State: Minimum of the c.v 
    Max_State: Maximum of the c.v
    Biasing_potential: Biasing potential in correct units (kcal/mol/A^2)
    Temp: Simulation temperature in kelvin
    sim_len: Length of SImulations
    """

    times = np.arange(Time_Step,sim_len+1, Time_Step)
    args = []
    pmf_results = []
    states = np.linspace(Min_State, Max_State, Bins)
    for i in times:
        cut = int(i)
        args.append([Bins, Trajectories[:, :cut], Bin_Centers, lag, Min_State, Max_State, Temp, Biasing_Potential])

    with multiprocessing.Pool(processes=int(os.cpu_count()/4)) as pool:
        results = pool.map(Parallel_Convergence, args)
        pmf_results.extend(results)

    pmf_results = np.asarray(pmf_results)

    AUC = []
    for count, i in enumerate(pmf_results):
        AUC.append(np.trapz(i, states))

    return(times, pmf_results, AUC)
