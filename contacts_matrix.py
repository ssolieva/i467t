# the following lists should be set up like: (wildtype, I467T)
top_files = ["/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories_wt/myh7-5n6a-holo-prot-masses.pdb",
             "/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories/myh7-5n6a-i467t-adp-phos-prot-masses.pdb"]
path_to_traj_lists = ["/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories_wt/traj_list.txt",
                      "/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories/traj_list.txt"]

import mdtraj as md
import numpy as np
import itertools
import matplotlib.pyplot as plt

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
    return atom_indices

def generate_binarized_contacts(traj_list, top_file, atom_indices, cutoff):
    '''this will load trajs, compute closest heavy atom distances,
    binarize the distances, and save the binarized distances '''
    dists_bin_all = []
    for i in traj_list:
        # load in trajectory with only certain residues
        trj = md.load(i, top=top_file, atom_indices=atom_indices)
        print(i, trj)
        # compute contacts
        ## create pairs using residue INDEX values for compute_contacts
        set1_ind = np.arange(0,len(set1))
        set2_ind = np.arange(len(set1),len(set1)+len(set2))
        pairs_ind = list(itertools.product(set1_ind, set2_ind))
        dists, pair_index = md.compute_contacts(trj,pairs_ind,scheme='closest-heavy')
        # binarize first, then save
        dists_bin_all.append((dists < cutoff).astype(np.bool_)) # cutoff in nm
    return dists_bin_all

def any_pairs(set1, set2, set1_name, set2_name,  top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name):
    # import traj_list
    with open(path_to_traj_list, "r") as fd:
        traj_list = fd.read().splitlines()
    # create pairs
    ## define set 1 and set 2, then combine the residue lists.
    pairs_res = np.concatenate([set1, set2]) # combined list
    # get atom indices
    atom_indices = generate_atom_ind(top_file, resis = pairs_res)
    # load in trajs, get binarized contacts
    dists_bin_all = generate_binarized_contacts(traj_list, top_file, atom_indices, cutoff=0.4)
    # save the binarized data
    np.save(f"{output_file1_name}",dists_bin_all)
    # take the average of the binarized data for each pair
    averaged_dists_bin = np.average(np.concatenate(dists_bin_all), axis=0)
    print(len(averaged_dists_bin)) #######NEW
    # save the averaged data
    np.save(f"{output_file2_name}",averaged_dists_bin)
    # reshape data into matrix form and then graph the matrix
    shape = ( len(set1), len(set2) )
    reshaped_dists = averaged_dists_bin.reshape( shape )
    fig, ax = plt.subplots()
    ax.matshow(reshaped_dists)
    for (i, j), z in np.ndenumerate(reshaped_dists):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',fontsize=8)
    plt.xticks(np.arange(len(set2)), set2)
    plt.yticks(np.arange(len(set1)), set1)
    plt.ylabel(f"{set1_name} (residue number)", fontsize=14)
    plt.xlabel(f"{set2_name} (residue number)", fontsize=14)
    plt.title(f"{matrix_name}")
    plt.savefig(f"{matrix_name}")

# WILDTYPE:
top_file = top_files[0]
path_to_traj_list = path_to_traj_lists[0]

set1_name = "S2"
set1 = np.arange(463,473)
set2_name = "relay helix"
set2 = np.arange(473,490)
output_file1_name = f"{set1_name}_{set2_name}_binarized_distances_wildtype"
output_file2_name = f"{set1_name}_{set2_name}_averaged_binarized_distances_wildtype"
matrix_name = f"{set1_name}_{set2_name}_wildtype"
any_pairs(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name)

set1_name = "S1"
set1 = np.arange(232,245)
set2_name = "S2"
set2 = np.arange(463,473)
output_file1_name = f"{set1_name}_{set2_name}_binarized_distances_wildtype"
output_file2_name = f"{set1_name}_{set2_name}_averaged_binarized_distances_wildtype"
matrix_name = f"{set1_name}_{set2_name}_wildtype"
any_pairs(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name)

# MUTANT:
top_file = top_files[1]
path_to_traj_list = path_to_traj_lists[1]

set1_name = "S2"
set1 = np.arange(463,473)
set2_name = "relay helix"
set2 = np.arange(473,490)
output_file1_name = f"{set1_name}_{set2_name}_binarized_distances_I467T"
output_file2_name = f"{set1_name}_{set2_name}_averaged_binarized_distances_I467T"
matrix_name = f"{set1_name}_{set2_name}_I467T"
any_pairs(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name)

set1_name = "S1"
set1 = np.arange(232,245)
set2_name = "S2"
set2 = np.arange(463,473)
output_file1_name = f"{set1_name}_{set2_name}_binarized_distances_I467T"
output_file2_name = f"{set1_name}_{set2_name}_averaged_binarized_distances_I467T"
matrix_name = f"{set1_name}_{set2_name}_I467T"
any_pairs(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name)