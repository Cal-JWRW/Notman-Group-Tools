import numpy as np
from DHAM import DHAM_Convergence
import multiprocessing
import argparse as ap
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

kj_kcal_conv = 0.239006
kt_kcal_conv = 0.592

def main():
    """
    Runs the bin convergence calculation utilizing multiprocessing to speed up calulation,
    produces a plot of bin convergence and the bin_convergence data 
    """
    os.system('mkdir ./DHAM_Analysis')
    while True:
        whoami = 'Script that calculates convergence criteria of markov state models constructed from biased umbrella sampling simulations \n'
        whoami += 'The script will calculate and return bin convergence criteria for markov models. \n'
        whoami += 'The bin convergence will return a plot from which to visually assess convergence point. \n'
       
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

        bp = (bp/100)*kj_kcal_conv

        break

    trj_tot = []
    cen_tot = []
    for j in tqdm(pxf):
        trj = np.abs(np.asarray(pd.read_table(j, header=None, skiprows=17, sep=r'\s+')[1]))*10
        cen_tot.append(trj[0])
        trj_tot.append(trj)
    trj_tot = np.asarray(trj_tot)

    
    bins, bins_pmf, AUC = DHAM_Convergence.Bin_Convergence(trj_tot, cen_tot, il, 1000, 10, mins, maxs, bp, temp, int(maxs))

    AUC = np.asarray(AUC)
    max_pmf = []
    for i in bins_pmf:
        max_pmf.append(max(i))

    fig, ax = plt.subplots()
    ax.plot(bins, AUC)
    ax.set_xlabel(r'$Bins$')
    ax.set_ylabel(r'$Area\;Under\;Curve$')
    fig.savefig('./DHAM_Analysis/Bin_AUC_Convergence.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(bins, max_pmf)
    ax.set_xlabel(r'$Bins$')
    ax.set_ylabel(r'$\Delta{}G$ / $\rm{}kJ\;mol^{-1}$')
    fig.savefig('./DHAM_Analysis/Bin_FE_Convergence.png', dpi=300)

    bin_data = pd.DataFrame({
        'Bins': bins,
        'AUC': AUC
    })
    bin_data.to_csv('./DHAM_Analysis/Bin_AUC_Data.csv', sep=' ', index=False)

    bin_data = pd.DataFrame({
        'Bins': bins,
        'Max_PMF': max_pmf
    })
    bin_data.to_csv('./DHAM_Analysis/Bin_PMF_Data.csv', sep=' ', index=False)

    return()

if __name__ == '__main__':
    main()