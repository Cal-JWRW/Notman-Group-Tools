import os
import numpy as np
import argparse as ap
import pandas as pd
import matplotlib.pyplot as plt

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

    """
    Following Inline Inputs:
    -px: .dat list of pullx files
    """

    while True:
        whoami = 'Script that calculates amount of data to remove from the beginning of the umbrella sampling simulations for equilibration \n'
        whoami += 'The script will calculate the autocorrelation functions of the C.V. data and calculate the largest time at which the a.c decays to 0 \n'
        parser = ap.ArgumentParser(description=whoami, formatter_class = ap.RawDescriptionHelpFormatter)
        parser.add_argument('-px','--px', nargs=1, type=str, help='List of pullx files')
        args = parser.parse_args()

        px = args.px[0]
        with open(px) as f:
            pxfiles = [line.rstrip() for line in f]

        os.system('mkdir ./DHAM_Analysis')
        break

    

    for count, i in enumerate(pxfiles):
        os.system(f'gmx analyze -f {i} -ac cv_ac.xvg')
        df = pd.read_table('cv_ac.xvg', header=None, skiprows=17, sep=r'\s+', engine='python')

        t=np.asarray(df[0])
        t=np.asarray(t[:-1], dtype=float)

        ac=np.asarray(df[1])
        ac = np.asarray(ac[:-1], dtype=float)
        
       
        eq_time = float(t[np.where(ac <= 0.0)[0][0]])

        if count == 0:
            print('FIRST')
            final_eq_time = eq_time
            final_ac = ac
            final_act = t

        if eq_time > final_eq_time:
            final_eq_time = eq_time
            final_ac = ac
            final_act = t

        print(eq_time)
        print(final_eq_time)


        os.system('rm cv_ac.xvg')

    fig, ax = plt.subplots()

    ax.plot(final_act, final_ac, linewidth=0.5, color='black')
    ax.axvline(x=final_eq_time, linestyle='--', color='red', label=f'{final_eq_time} ps')
    ax.set_xlabel(r'$Time$ / $\rm{}ps$')
    ax.set_ylabel(r'$C(z)$')
    ax.legend()

    print('SAVING')
    fig.savefig('./DHAM_Analysis/AC_Equil.png', dpi=300)

    return

if __name__ == '__main__':
    main()