import lipyphilic as lpp
import MDAnalysis as mda
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import os


plt.rcParams["figure.figsize"] = (9,7)
STANDARD_SIZE=18
SMALL_SIZE=14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=STANDARD_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=STANDARD_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=STANDARD_SIZE)    # legend fontsize

def main():
    '''
    Required Inputs :
    xtc: .dat file containing xtc file names
    tpr: tpr file name
    A: Atom A name
    B: Atom B Name 
    RI: Residue Index of Permeant
    '''
    while True:

        whoami = 'This script calculates the angle distribution of a permeant through the stratum corneum'

        parser = ap.ArgumentParser(description = whoami, formatter_class = ap.RawDescriptionHelpFormatter)
        parser.add_argument('-xtc', '--xtc', required=True, nargs=1, type=str, help='.dat file containing xtc filenames')
        parser.add_argument('-px', '--px', required=True, nargs=1, type=str, help='.dat file containing pullx filenames')
        parser.add_argument('-tpr', '--tpr', required=True, nargs=1, type=str, help='tpr file name')
        parser.add_argument('-A', '--A', required=True, nargs=1, type=str, help='Name of Atom A (Head of vector)')
        parser.add_argument('-B', '--B', required=True, nargs=1, type=str, help='Name of Atom B (Tail of vector)')
        args = parser.parse_args()

        xtc = args.xtc[0]
        px = args.px[0]
        tpr = args.tpr[0]
        atomA = args.A[0]
        atomB = args.B[0]

        with open(xtc) as f:
            xtcfiles = [line.rstrip() for line in f]

        with open(px) as f:
            pxfiles = [line.rstrip() for line in f]

        os.system('mkdir ./Perm_Angle')
 
        break


    Angles = []
    COM_Pos = []

    for i in tqdm(pxfiles, desc='Reading pullx files...'):

        df = pd.read_table(i, skiprows=17, header=None, sep=r'\s+')
        COM_Pos.append(int(round(df[1][0]*10)))

    for i in tqdm(xtcfiles, desc='Calculating z angles...'):
        
        u = mda.Universe(tpr, i)

        residues = u.residues
        resids = [residue.resid for residue in residues]
        resid = resids[-1]


        z_angles = lpp.analysis.ZAngles(
            universe = u,
            atom_A_sel=f'name {atomA} and resid {resid}',
            atom_B_sel=f'name {atomB} and resid {resid}'
        )
        z_angles.run(start=None, stop=None, step=None)

        Angles.append(np.abs(z_angles.z_angles[0]))   

    Probabilities = []

    Angles_Bins = np.arange(0.0, 181.0, 5)
    
    for i in Angles:
        Angles_Digitized = (np.digitize(i, Angles_Bins, right=True)-1)

        Count = np.zeros(len(Angles_Bins), dtype=float)
        for j in Angles_Digitized:
            Count[j] += 1

        Count = Count/sum(Count)
        Probabilities.append(Count)

    Probabilities = np.asarray(Probabilities)

    combined = list(zip(COM_Pos, Probabilities))

    combined.sort(key=lambda x: x[0])
    
    # Unzip the combined list back into two sorted lists
    COM_Pos, Probabilities = zip(*combined)

    COM_Pos = np.asarray(COM_Pos)
    Probabilities = np.asarray(Probabilities, dtype=float)

    fig, ax  =plt.subplots()
    ax = sns.heatmap(Probabilities.T, xticklabels=COM_Pos, yticklabels=Angles_Bins, cmap="rocket_r", vmin=0, vmax=0.5)
    ax.set_xticks(ax.get_xticks()[::4])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_ylabel(r'$Permeant\;Angle$ / $\rm{}^{o}$')
    ax.set_xlabel(r'$z$ / $\rm{}\AA{}$')

    cbar = ax.collections[0].colorbar
    cbar.set_label('Probability', rotation=270, labelpad=20)

    fig.savefig('./Perm_Angle/Perm_Angle_Sorted.png', dpi=300)

    np.save('./Perm_Angle/Probabilities.npy', Probabilities)
    np.save('./Perm_Angle/Angle_Bins.npy', Angles_Bins)
    np.save('./Perm_Angle/COM_Positions.npy', COM_Pos)
    




    




    return()

if __name__ == '__main__':
    main()