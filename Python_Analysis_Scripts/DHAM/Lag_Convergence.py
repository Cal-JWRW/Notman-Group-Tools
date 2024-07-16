import numpy as np
from DHAM import DHAM_Lagtime
from DHAM import DHAM_Convergence
import multiprocessing
import argparse as ap
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

kj_kcal_conv = 0.239006
kt_kcal_conv = 0.592

def running_average(data, window_size):
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer")


    running_averages = []
    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        current_window = data[start_index:i+1]
        window_average = sum(current_window) / len(current_window)
        running_averages.append(window_average)
    
    return running_averages

def main():
    """
    Runs the bin convergence calculation utilizing multiprocessing to speed up calulation,
    produces a plot of bin convergence and the bin_convergence data 
    """
    os.system('mkdir ./DHAM_Analysis')
    while True:
        whoami = 'Script that calculates convergence criteria of markov state models constructed from biased umbrella sampling simulations \n'
        whoami += 'The script will calculate and return lagtime and time convergence criteria for markov models. \n'
    
        whoami += 'The lagtime convergence will return two plots featuring the relaxation time and first differential to assess convergence. \n'
        whoami += 'The time convergence calculates the profile as we include more data, plots the maxima of the free energy and then calculates the variance as we remove data. \n'
        parser = ap.ArgumentParser(description=whoami, formatter_class = ap.RawDescriptionHelpFormatter)
        parser.add_argument('-px','--px', nargs=1, type=str, help='List of pullx files')
        parser.add_argument('-bp','--bp', nargs=1, type=int, help='Umbrella biasing potential in kJ/mol/nm^2')
        parser.add_argument('-ib', '--ib', nargs=1, type=int, help='Ideal Bin Number' )
        parser.add_argument('-temp', '--temp', type=float, nargs=1, help='Simulation temperature in kelvin')
        parser.add_argument('-mins','--mins', nargs=1, type=float, help='Minimum of the umbrella collective variable in angstroms')
        parser.add_argument('-maxs', '--maxs', nargs=1, type=float, help='Maximum of the collective variable in angstroms')
        

        #Extract Inputs
        args = parser.parse_args()

        px = args.px[0]
        bp = args.bp[0]
        ib = args.ib[0]
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

    
    MMs = DHAM_Lagtime.Relaxation_Times(trj_tot, cen_tot, ib, mins, maxs, bp, temp, 1, 5000, 10)

    vs = []
    t = np.arange(1, 5001, 10)*1E-13
    for count, i in enumerate(MMs):
        sl = sorted(set(i))[-2] if len(set(i)) > 1 else None
        vs.append(-t[count]/np.log(sl))
    vs = np.asarray(vs).astype(np.float128)


    logvs = np.log10(vs)
    vs_grad = np.diff(logvs)/np.diff(t)
    vs_grad = running_average(vs_grad, int(len(vs_grad)/10))
    vs_grad = np.asarray(vs_grad)
    vs_grad = (vs_grad-min(vs_grad))/(max(vs_grad)-min(vs_grad))
    plat_end = int(np.where(vs_grad <= np.mean(vs_grad))[0][1])

    fig, ax = plt.subplots()
    ax.plot(t, vs)
    ax.set_yscale('log')
    ax.ticklabel_format(axis='x', style='sci')
    ax.set_xlim([0, 5E-10])
    ax.set_ylim([1E-10, 1E-0])
    ax.set_xlabel(r'$Lagtime$ / $\rm{}s$')
    ax.set_ylabel(r'$Implied\;Relaxation\;Times$ / $\rm{} s$')
 
    ideal_lagtime = int(t[plat_end]*1E13)

    ax.axvline(x=t[plat_end], color='r', linestyle='--', label=f'Ideal Lagtime: {ideal_lagtime} x10^-13 s')
    
    ax.legend()
    fig.savefig('./DHAM_Analysis/Lag_Convergence.png', dpi=300)

    lag_data = pd.DataFrame({
        'Time': t,
        'Implied Relaxation Time': vs
    })
    lag_data.to_csv('./DHAM_Analysis/Lag_Data.csv', sep=' ', index=False)
    

    Equil_Times, Equil_PMF, Equil_AUC = DHAM_Convergence.Time_Equilibration(trj_tot, cen_tot, ideal_lagtime,ib, 10000, mins, maxs, bp, temp, int(len(trj_tot[0])+1))

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

    fig.savefig('./DHAM_Analysis/Time_Equilibration.png', dpi=300)
    
    time_data = pd.DataFrame({
        'Time': Equil_Times,
        'MaxFE': maxfe
    })
    time_data.to_csv('./DHAM_Analysis/Time_Data.csv', sep=' ', index=False)


    fig, ax = plt.subplots()
    colors = plt.cm.jet(np.linspace(0,1,len(Equil_Times)))
    plt_states = np.linspace(mins, maxs, ib)

    for i in range(len(Equil_Times)):
        ax.plot(plt_states, Equil_PMF[i]/kj_kcal_conv, color=colors[i])

    # Add a colorbar to indicate the mapping of colors to times
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=min(Equil_Times), vmax=max(Equil_Times)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'$Time$ / $\rm{} Datapoints$')

    tick_locator = plt.MaxNLocator(nbins=10)  # Adjust nbins to control the number of ticks
    cbar.locator = tick_locator
    cbar.update_ticks()

    ax.set_xlabel(r'$z$ / $\rm{} \AA{}$')
    ax.set_ylabel(r'$\Delta{}G$ / $\rm{} kJ\;mol^{-1}$')

    fig.tight_layout()
    fig.savefig('./DHAM_Analysis/Time_Convergence.png', dpi=300)

    fig, ax  = plt.subplots()
    ax.plot(Equil_Times, maxfe, linestyle='--', marker='o', color='red', markerfacecolor='black', markeredgecolor='black')
    ax.set_xlabel(r'$Simulation\;Steps')
    ax.set_ylabel(r'$Maximum\;Free\;Energy$ / $\rm{}kJ\;mol^{-1}$')
    fig.savefig('./DHAM_Analysis/Time_Convergence_MaxFE.png', dpi=300)


    return()

if __name__ == '__main__':
    main()