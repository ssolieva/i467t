import mdtraj as md
import numpy as np
import itertools
import matplotlib.pyplot as plt

# name the output file
output_file1_name = "binarized_distances_wildtype"
output_file2_name = "averaged_binarized_distances_wildtype"
matrix_name = "matrix_avg_bin_dists_wildtype"

# paths
top_file = '/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories_wt/myh7-5n6a-holo-prot-masses.pdb'
path_to_traj_list = '/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories_wt/traj_list.txt'

# import traj_list
with open(path_to_traj_list, "r") as fd:
    traj_list = fd.read().splitlines()

def generate_atom_ind(top_file, resis, stride=1, include_ADP=False, include_PO4=False):
    '''this will take a top_file + a list of residues and output atom indices for those residues.
        resis is a list of residue numbers'''
    topology = md.load(top_file)
    selection_string = ' or '.join(f'residue {r}' for r in resis)
    if include_ADP:
        selection_string += ' or resname ADP'
        # append resSeq of ADP
        resis.append(topology.top.atom(topology.top.select('resname ADP')[0]).residue.resSeq)
    if include_PO4:
        selection_string += ' or resname PO4'
        # append resSeq of PO4
        resis.append(topology.top.atom(topology.top.select('resname PO4')[0]).residue.resSeq)
    atom_indices = topology.top.select(selection_string)
    atom_indices_saved.append(atom_indices)    

def generate_binarized_contacts(traj_list, top_file, atom_indices, cutoff):
    '''this will load trajs, compute closest heavy atom distances, 
    binarize the distances, and save the binarized distances '''
    for i in traj_list:
        # load in trajectory with only certain residues 
        trj = md.load(i, top=top_file, atom_indices=atom_indices)
        print(i, trj)
        # compute contacts
        dists, pair_index = md.compute_contacts(trj,pairs_ind,scheme='closest-heavy')
        # binarize first, then save
        dists_bin_all.append((dists < cutoff).astype(np.int_))# cutoff in nm 
        pair_index_all.append(pair_index)

print('test')

# create pairs
## define switch 1 and switch 2, then combine the residue lists.
switch_1 = np.arange(463,473) # res 463-472
switch_2 = np.arange(232,245) # res 232-244
pairs_res = np.concatenate([switch_2, switch_1]) # combined list

## create pairs using residue INDEX values for compute_contacts
switch2_ind = np.arange(0,len(switch_2))
switch1_ind = np.arange(len(switch_2),len(switch_1)+len(switch_2))
pairs_ind = list(itertools.product(switch2_ind, switch1_ind))

# save out atom indices
atom_indices_saved = []  
generate_atom_ind(top_file, resis = pairs_res)

# load in trajs, get binarized contacts
dists_bin_all = []
pair_index_all = []
generate_binarized_contacts(traj_list, top_file, atom_indices = atom_indices_saved[0], cutoff = 0.4)

# save the binarized data
np.save("%s"%output_file1_name,dists_bin_all)

# take the average of the binarized data for each pair
averaged_dists_bin = np.average(np.concatenate(dists_bin_all), axis=0)

# save the averaged data
np.save("%s"%output_file2_name,averaged_dists_bin)

# reshape data into matrix form and then graph the matrix
shape = ( 13, 10 )
reshaped_dists = averaged_dists_bin.reshape( shape )

fig, ax = plt.subplots(figsize=(5,6))
ax.matshow(reshaped_dists)
for (i, j), z in np.ndenumerate(reshaped_dists):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',fontsize=8)
plt.xticks(np.arange(10), np.arange(463,473))
plt.yticks(np.arange(13), np.arange(232,245))
plt.ylabel("S2 (residue number)", fontsize=14)
plt.xlabel("S1 (residue number)", fontsize=14)
plt.savefig("%s"%matrix_name)
