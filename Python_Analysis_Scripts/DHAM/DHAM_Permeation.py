from DHAM import DHAM
from DHAM import Build_Matrices
from DHAM import DHAM_Diffusion
import numpy as np
import argparse as ap
import os
from tqdm import tqdm
import pandas as pd
from scipy.linalg import eig
import matplotlib.pyplot as plt

kj_kcal_conv = 0.239006
kt_kcal_conv = 0.592
R_kcal_mol = (1.987/1000)

def main():
    """
    Following Inline Inputs:
    -px: .dat list of pullx files
    -bp: Biasing force in kJ/mol/nm^2
    -ib: Ideal number of bins
    -il: Ideal lagtime in pullx steps
    -temp: Simulation temperature in kelvin
    -mins: Minimum value of the collective variable
    -maxs Maximum value of the collective variable

    """
    os.system('mkdir ./DHAM_Analysis')
    while True:
        whoami = 'Script that calculates small molecule membrane permeation rates from biased umbrella sampling simulations using DHAM \n'
        whoami += 'The script will calculate the PMF, Diffusion profile and permeation rate \n'
        parser = ap.ArgumentParser(description=whoami, formatter_class = ap.RawDescriptionHelpFormatter)
        parser.add_argument('-px','--px', nargs=1, type=str, help='List of pullx files')
        parser.add_argument('-bp','--bp', nargs=1, type=int, help='Umbrella biasing potential in kJ/mol/nm^2')
        parser.add_argument('-ib', '--ib', nargs=1, type=int, help='Ideal Bin Number' )
        parser.add_argument('-il', '--il', nargs=1, type=int, help='Ideal Lag Time (Simulation Steps)' )
        parser.add_argument('-ic', '--ic', nargs=1, type=int, help='Ideal Cutoff (Simulation Steps)' )
        parser.add_argument('-temp', '--temp', type=float, nargs=1, help='Simulation temperature in kelvin')
        parser.add_argument('-mins','--mins', nargs=1, type=float, help='Minimum of the umbrella collective variable in angstroms')
        parser.add_argument('-maxs', '--maxs', nargs=1, type=float, help='Maximum of the collective variable in angstroms')
        
        

        #Extract Inputs
        args = parser.parse_args()

        px = args.px[0]
        bp = args.bp[0]
        ib = args.ib[0]
        il = args.il[0]
        ic = args.ic[0]
        temp = args.temp[0]
        mins = args.mins[0]
        maxs = args.maxs[0]

        bp = (bp/100)*kj_kcal_conv

        with open(px) as f:
            pxf = [line.rstrip() for line in f]

        states = np.linspace(mins, maxs, ib)

        break

    cen_tot = []
    trj_tot = []

    for j in tqdm(pxf, desc='Extracting Trajectories...'):
        trj = np.abs(np.asarray(pd.read_table(j, skiprows=17, sep=r'\s+', header=None)[1]))*10
        trj_tot.append(trj[ic:])
        cen_tot.append(trj[0])

    c_ij = []
    bias_i = []

    for j in tqdm(range(len(trj_tot)), desc='Calculating Transition and Bias Matrices'):
        c_ij.append(Build_Matrices.Count_Matrix(trj_tot[j], il, states))
        bias_i.append(Build_Matrices.Bias_Matrix(trj_tot[j], bp, states, cen_tot[j]))

    c_ij = np.asarray(c_ij)
    bias_i = np.asarray(bias_i)

    MM = DHAM.Construct_MM(c_ij, bias_i.T, temp)
    d, v = eig(MM.T)
    mpeq = v[:, np.where(d == np.max(d))[0][0]]
    mpeq = mpeq / np.sum(mpeq)
    mU2 = -temp * R_kcal_mol * np.log(mpeq)
    mU2[-10:] = mU2[-20:-10]
    mU2 -= np.min(mU2[:int(len(states))])

    Diff_Coeffs = DHAM_Diffusion.Calculate_Diffusion(MM, states, il)
    Resistivity = np.exp(mU2/(temp*R_kcal_mol))/Diff_Coeffs

    perm = 1/(np.trapz(Resistivity, states)*10**-8)

    tot_states = []
    tot_pmf = []
    tot_diff = []

    for i in range(len(states)-1, -1, -1):
        tot_states.append(-states[i])
        tot_pmf.append(mU2[i])
        tot_diff.append(Diff_Coeffs[i])

    for i in range(0, len(states)):
        tot_states.append(states[i])
        tot_pmf.append(mU2[i])
        tot_diff.append(Diff_Coeffs[i])

    tot_states = np.asarray(tot_states)
    tot_pmf = np.asarray(tot_pmf)
    tot_diff = np.asarray(tot_diff)

    fig, ax = plt.subplots()
    ax.plot(tot_states, tot_pmf/kj_kcal_conv)
    ax.set_xlabel(r'$z$ / $\rm{}\AA{}$')
    ax.set_ylabel(r'$\Delta{}G$ / $\rm{}kJ\;mol^{-1}$')
    fig.savefig('./DHAM_Analysis/Final_PMF.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(tot_states, tot_diff, linestyle='--', marker='o', color='black', markerfacecolor='red', markeredgecolor='red', markersize=1)
    ax.set_yscale('log')
    ax.set_xlabel(r'$z$ / $\rm{}\AA{}$')
    ax.set_ylabel(r'$Diffusion$ / $\rm{}cm^{2}\;s^{-1}$')
    fig.savefig('./DHAM_Analysis/Diffusion.png', dpi=300)

    np.save('./DHAM_Analysis/PMF.npy', mU2/kj_kcal_conv)
    np.save('./DHAM_Analysis/DIFF.npy', Diff_Coeffs)
    np.save('./DHAM_Analysis/STATES.npy', states)

    print(f"Permeation Coefficient: {perm} cm^2/s")
    print(f"Permeation Coefficient: {perm*3600} cm^2/hr")

    with open("./DHAM_Analysis/Perm_Coeff.txt", "w") as file:
        # Write the line of text to the file
        file.write(f"Permeation Coefficient: {perm} cm^2/s")
        file.write(f"Permeation Coefficient: {perm*3600} cm^2/hr")


    return

if __name__ == '__main__':
    main()





