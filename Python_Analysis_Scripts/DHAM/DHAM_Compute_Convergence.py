from DHAM import DHAM_Convergence
from DHAM import DHAM_Lagtime
import argparse as ap
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np 

plt.rcParams["figure.figsize"] = (9,7)
STANDARD_SIZE=18
SMALL_SIZE=14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=STANDARD_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=STANDARD_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=STANDARD_SIZE)    # legend fontsize

kj_kcal_conv = 0.239006
kt_kcal_conv = 0.592

import numpy as np
import matplotlib.pyplot as plt

def find_bin_plateau_end(data, window_size=5, std_threshold=1.5):
    """
    Identify the end of the plateau in the bin convergence

    Parametrs: 
        data (list or np.array): The input data set
        window_size (int): The size of sliding window
        std_threshold (float): The threshold for the standard deviation under which to consider a plateau
    
    Returns:
        int: The index of the plateau
    """

    data = np.array(data)

    #Calulate rolling standard deviation
    rolling_std = np.array([np.std(data[i:i+window_size]) for i in range(len(data)-window_size+1)])
    rolling_std = rolling_std/(min(rolling_std)+0.1) #Normalize  standard deviation

    plateau_end_indices = np.where(rolling_std < std_threshold)[0]

    #Return the last index where the standard deviation is below  the threshold
    if len(plateau_end_indices) > 0:
        return (plateau_end_indices[-1] + window_size-1)
    else: 
        return len(data)-1


def find_lag_plateau_start(data, window_size=3, std_threshold=0.5):
    
    """
    Identify the end of a plateau in the data set.
    
    Parameters:
        data (list or np.array): The input data set.
        window_size (int): The size of the sliding window.
        std_threshold (float): The threshold for the standard deviation to consider as plateau.
    
    Returns:
        int: The index of the end of the plateau.
    """
    data = np.array(data)
    
    # Calculate the rolling standard deviation
    rolling_std = np.array([np.std(data[i:i+window_size]) for i in range(len(data) - window_size + 1)])
    rolling_std = rolling_std/(min(rolling_std)+0.01)
    print(rolling_std)
    
    # Identify where the standard deviation exceeds the threshold
    plateau_end_indices = np.where(rolling_std < std_threshold)[0]
    
    # Return the first index where the standard deviation exceeds the threshold
    if len(plateau_end_indices) > 0:
        return plateau_end_indices[0] + window_size - 1
    else:
        return len(data) - 1  # If no plateau end is found, return the last index
    
def main():
    os.system('mkdir ./DHAM_Analysis')
    while True:
        whoami = 'Script that calculates convergence criteria of markov state models constructed from biased umbrella sampling simulations \n'
        whoami += 'The script will calculate and return bin, lagtime and time convergence criteria for markov models. \n'
        whoami += 'The bin convergence will return a plot from which to visually assess convergence point. \n'
        whoami += 'The lagtime convergence will return two plots featuring the relaxation time and first differential to assess convergence. \n'
        whoami += 'The time convergence calculates the profile as we include more data, plots the maxima of the free energy and then calculates the variance as we remove data. \n'
        parser = ap.ArgumentParser(description=whoami, formatter_class = ap.RawDescriptionHelpFormatter)
        parser.add_argument('-px','--px', nargs=1, type=str, help='List of pullx files')
        parser.add_argument('-bp','--bp', nargs=1, type=int, help='Umbrella biasing potential in kJ/mol/nm^2')
        parser.add_argument('-il', '--il', nargs=1, type=int, help='Initial lagtime estimate' )
        parser.add_argument('-temp', '--temp', type=float, nargs=1, help='Simulation temperature in kelvin')
        parser.add_argument('-mins','--mins', nargs=1, type=float, help='Minimum of the umbrella collective variable in angstroms')
        parser.add_argument('-maxs', '--maxs', nargs=1, type=float, help='Maximum of the collective variable in angstroms')

        #Extract Inputs
        args = parser.parse_args()

        px = args.px[0]
        bp = args.bp[0]
        il = args.il[0]
        temp = args.temp[0]
        mins = args.mins[0]
        maxs = args.maxs[0]

        with open(px) as f:
            pxf = [line.rstrip() for line in f]

        break
    
    bp = (bp/100)*kj_kcal_conv

    trj_tot = []
    cen_tot = []
    for j in tqdm(pxf):
        trj = np.abs(np.asarray(pd.read_table(j, header=None, skiprows=17, sep=r'\s+')[1]))*10
        cen_tot.append(trj[0])
        trj_tot.append(trj)
    trj_tot = np.asarray(trj_tot)

    bins, bins_pmf, AUC = DHAM_Convergence.Bin_Convergence(trj_tot, cen_tot, il, 1000, 10, mins, maxs, bp, temp, int(maxs))
    bin_plateau_end = find_bin_plateau_end(AUC, (1000/(10*10)), 0.5)
    ideal_bins = bins[bin_plateau_end]

    fig, ax = plt.subplots()
    ax.plot(bins, AUC)
    ax.axvline(x=ideal_bins, color='r', linestyle='--', label=f'Ideal Bin No: {ideal_bins}')
    ax.set_xlabel(r'$Bins$')
    ax.set_ylabel(r'$Area\;Under\;Curve$')
    ax.legend()
    fig.savefig('./DHAM_Analysis/Bin_Convergence.png', dpi=300)

    bin_data = pd.DataFrame({
        'Bins': bins,
        'AUC': AUC
    })
    bin_data.to_csv('./DHAM_Analysis/Bin_Data.csv', sep=' ', index=False)

    MMs = DHAM_Lagtime.Relaxation_Times(trj_tot, cen_tot, ideal_bins, mins, maxs, bp, temp, 1, 5000, 10)

    vs = []
    t = np.arange(1, 5001, 10)*1E-13
    for count, i in enumerate(MMs):
        sl = sorted(set(i))[-2] if len(set(i)) > 1 else None
        vs.append(-t[count]/np.log(sl))
    vs = np.asarray(vs).astype(np.float128)

    fig, ax = plt.subplots()
    ax.plot(t, vs)
    ax.set_yscale('log')
    ax.ticklabel_format(axis='x', style='sci')
    ax.set_xlim([0, 5E-10])
    ax.set_ylim([1E-10, 1E-0])
    ax.set_xlabel(r'$Lagtime$ / $\rm{}s$')
    ax.set_ylabel(r'$Implied\;Relaxation\;Times$ / $\rm{} s$')
    time_plateau_start = find_lag_plateau_start(np.diff(vs, 2))
    ideal_lagtime = int(t[time_plateau_start]*1E13)

    ax.axvline(x=t[time_plateau_start], color='r', linestyle='--', label=f'Ideal Lagtime: {ideal_lagtime} x10^-13 s')
    
    ax.legend()
    fig.savefig('./DHAM_Analysis/Lag_Convergence.png', dpi=300)

    lag_data = pd.DataFrame({
        'Lags': t,
        'Relaxation': vs
    })
    lag_data.to_csv('./DHAM_Analysis/Lag_Data.csv', sep=' ', index=False)

    Equil_Times, Equil_PMF, Equil_AUC = DHAM_Convergence.Time_Equilibration(trj_tot, cen_tot, ideal_lagtime,ideal_bins, 10000, mins, maxs, bp, temp, 1400001)

    
    maxfe = [max(a) for a in Equil_PMF]
    mean = np.mean(maxfe)
    var=[]
    for i in range(0, len(maxfe)):
        variance = np.sum([(a-mean)**2 for a in maxfe[i:]])/(len(maxfe[i:])-1)
        var.append(variance)

    fig, ax = plt.subplots()
    ax.plot(Equil_Times, var)
    ax.set_xlabel(r'$Time\;Removed$ / $\rm{}Steps$')
    ax.set_ylabel(r'$Variance$')

    ideal_cutoff = Equil_Times[np.argmin(var)]
    ax.axvline(x=ideal_cutoff, color='r', linestyle='--', label=f'Ideal cutoff: {ideal_cutoff}')
    ax.legend()

    fig.savefig('./DHAM_Analysis/Time_Convergence.png', dpi=300)
    
    time_data = pd.DataFrame({
        'Time': Equil_Times,
        'MaxFE': maxfe
    })
    time_data.to_csv('./DHAM_Analysis/Time_Data.csv', sep=' ', index=False)
    

    return()

if __name__ == '__main__':
    main()
    print('here')