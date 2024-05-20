import pymol
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_itp_file(itp_file):
    charges = {}
    with open(itp_file, 'r') as file:
        lines = file.readlines()
        read_atomtypes = False
        for line in lines:
            if line.startswith('[ atoms ]'):
                read_atomtypes = True
                continue
            if read_atomtypes:
                if line.startswith('[') or line.strip() == '':
                    break
                if line.strip() and not line.startswith(';'):
                    parts = line.split()
                    atom_name = parts[4]
                    partial_charge = float(parts[6])
                    charges[atom_name] = partial_charge
    return charges

   


def color_by_partial_charge(itp_files, pdb_file, object_sel, residue_selection):
    # Read partial charges from the .itp file
    itp_files = itp_files.split()
    print(itp_files)    
    
    residue_selection = residue_selection.split()
    print(residue_selection)

    atom_charges = []
    atom_indices = []
    for count, itp_file in enumerate(itp_files):
        charges = read_itp_file(itp_file)
        selection = object_sel + ' and resn ' + residue_selection[count]
        print(selection)
      
        # Iterate over atoms in the selection and assign charges
        model = cmd.get_model(selection)
        for atom in model.atom:     
            atom_name = atom.name
            for keys in charges:            
                if atom_name[:len(keys)] == keys:
                    atom_charges.append(charges[keys])
                    atom_indices.append(atom.index)
    atom_charges = np.array(atom_charges)
    
    # Normalize charges to range [0, 1]
    min_charge = -1
    max_charge = 1
    norm_charges = (atom_charges - min_charge) / (max_charge - min_charge)
    
    # Define a color gradient
    colormap = plt.get_cmap('coolwarm')
    colors = colormap(norm_charges)
    
    # Apply colors to atoms
    for i, color in tqdm(enumerate(colors), desc='Coloring...'):
        color_name = f'color_{i}'
        cmd.set_color(color_name, list(color[:3]))  # PyMOL expects RGB values
        cmd.color(color_name, f'({object_sel}) and index {atom_indices[i]}')

cmd.extend("ColByChg", color_by_partial_charge)
