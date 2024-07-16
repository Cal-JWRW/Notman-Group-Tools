#Imports
import os,sys,argparse,subprocess
import numpy as np
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

def calc_hbonds(xtc, tpr, index, ig1, ig2):
    ave_hbonds= 0
    cmd = f'gmx hbond -f {xtc} -s {tpr} -n {index} -b 100000 -num ./HB_TM/{xtc[:-4]}.xvg<<EOF\n{ig1}\n{ig2}\nEOF'
    process = subprocess.run(cmd, shell=True)
    df = pd.read_table(f'./HB_TM/{xtc[:-4]}.xvg', skiprows=25, header=None, delim_whitespace=True)
    ave_hbonds = np.mean(df[1])
    return(ave_hbonds)

def extract_z(px):
    return(pd.read_table(f'{px}', skiprows=17, header=None, delim_whitespace=True)[1][0]*10)

def main():
    while True:
        whoami = "This script takes a list of xtc and tpr files and calcul;ates the number of hydrogen bonds a \n"
        whoami += "Permeant makes with its surrounding molecules thoughout a membrane"
        parser = argparse.ArgumentParser(description = whoami, formatter_class = argparse.RawDescriptionHelpFormatter)

        parser.add_argument('-xtc','--xtc', type=str, nargs=1, help='.dat file containing xtc file names')
        parser.add_argument('-px','--px', type=str, nargs=1, help='.dat file containing pullx file names')
        parser.add_argument('-tpr','--tpr', type=str, nargs=1, help='Name of tpr file')
        parser.add_argument('-n','--n', type=str, nargs=1, help='Name of index file')
        parser.add_argument('-perm','--name1', type=str, nargs=1, help='name of index group containing permeant molecule only')
        parser.add_argument('-sys','--name2', type=str, nargs=1, help='name of index group containing mmebrane molecules only')


        args=parser.parse_args()

        xtc_file = args.xtc[0]
        px_file = args.px[0]
        tpr = args.tpr[0]
        index = args.n[0]
        perm_group = args.name1[0]
        memb_group = args.name2[0]

        with open(xtc_file) as f:
            xtc = [line.rstrip() for line in f]

        with open(px_file) as f:
            px = [line.rstrip() for line in f]

        os.system('mkdir HB_TM')
        break

    ave_hb = np.zeros(len(xtc))
    z = np.zeros(len(xtc))

    for count, f in enumerate(xtc):
        ave_hb[count] = calc_hbonds(f, tpr, index, perm_group, memb_group)
        z[count] = extract_z(px[count])

    fig, ax1 = plt.subplots()
    ax1.scatter(z, ave_hb)
    ax1.set_xlabel(r'$z$/$\rm{} \AA$')
    ax1.set_ylabel(r'$Ave\;Hbonds$')

    plt.savefig('./HB_TM/num.png', dpi=300)
    plt.show()

    data = np.vstack((z, ave_hb))
    np.save('./HB_TM/data.npy', data)

    return()

if __name__ == '__main__':
    main()