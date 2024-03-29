import mdtraj as md
import numpy as np
import itertools
import matplotlib.pyplot as plt

# the following lists should be set up like: (wildtype, I467T)
top_files = ["/home/artur/bowmore-backup/i467t/wt-trajectories/myh7-5n6a-holo-prot-masses.pdb",
             "/home/artur/bowmore-backup/i467t/i467t-trajectories/myh7-5n6a-i467t-adp-phos-prot-masses.pdb"]
path_to_traj_lists = ["wt-trajectories.txt",
                      "i467t-trajectories.txt"]

def generate_neighbor_atom_ind(top_file, query_residue, cutoff=0.5):
    '''this will take a top_file + one residue and output neighboring atom indices for that residues.
        resis is a list of residue numbers'''
    topology = md.load(top_file)
    query_indices = topology.top.select(f'residue {query_residue}')
    haystack_indices = topology.top.select('protein')
    atom_indices = md.compute_neighbors(topology,cutoff, query_indices, haystack_indices)[0]
    unique_resis = np.unique(
        [topology.top.atom(ai).residue.resSeq for ai in atom_indices if topology.top.atom(ai).residue.resSeq != 466 and topology.top.atom(ai).residue.resSeq != 468]
    )
    # remove neighbors
    #unique_resis = [r for r in unique_resis if np.abs(r - query_residue) > 1]
    selection_string = ' or '.join(f'residue {r}' for r in unique_resis)
    atom_indices = topology.top.select(selection_string)
    return atom_indices, unique_resis

def generate_neighbor_binarized_contacts(traj_list, top_file, atom_indices, index_value_467, unique_resis,cutoff=0.4): ########
    '''this will load trajs, compute closest heavy atom distances,
    binarize the distances, and save the binarized distances '''
    dists_bin_all = []
    for i in traj_list:
        # load in trajectory with only certain residues
        trj = md.load(i, top=top_file, atom_indices=atom_indices)
        print(i, trj)
        # compute contacts
        ## create pairs using residue INDEX values for compute_contacts
        set1_ind = index_value_467
        set2_ind = np.arange(0,len(unique_resis))
        pairs_ind = [(a, b) for a in set1_ind for b in set2_ind if a != b]
        #print(pairs_ind)
        dists, pair_index = md.compute_contacts(trj,pairs_ind,scheme='closest-heavy')
        # binarize first, then save
        #print(pair_index)
        dists_bin_all.append((dists < cutoff).astype(np.bool_)) # cutoff in nm
    return dists_bin_all

def neighbors_wt(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name):
    assert len(set1) == 1, 'set 1 is more than 1 residue, we should only be computing neighbors'
    # import traj_list
    with open(path_to_traj_list, "r") as fd:
        traj_list = fd.read().splitlines()
    # get atom indices
    atom_indices,unique_resis= generate_neighbor_atom_ind(top_file, set1[0])
    index_value_467 = []
    for i in range(len(unique_resis)):
        if unique_resis[i] == 467:
            print("index value:",i)
            index_value_467.append(i)
    # load in trajs, get binarized contacts
    dists_bin_all = generate_neighbor_binarized_contacts(traj_list, top_file, atom_indices, index_value_467, unique_resis)
    # save the binarized data
    np.save(f"{output_file1_name}",dists_bin_all)
    # take the average of the binarized data for each pair
    averaged_dists_bin = np.average(np.concatenate(dists_bin_all), axis=0)
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
    plt.ylabel(f"{set1_name}", fontsize=10)
    plt.xlabel(f"{set2_name} (residue number)", fontsize=10)
    plt.title(f"{matrix_name}")
    plt.tight_layout()
    plt.savefig(f"{matrix_name}")

def neighbors_mt(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name, top_file_wt):
    assert len(set1) == 1, 'set 1 is more than 1 residue, we should only be computing neighbors'
    # import traj_list
    with open(path_to_traj_list, "r") as fd:
        traj_list = fd.read().splitlines()
    # get atom indices
    atom_indices,unique_resis= generate_neighbor_atom_ind(top_file_wt, set1[0])
    index_value_467 = []
    for i in range(len(unique_resis)):
        if unique_resis[i] == 467:
            print("index value:",i)
            index_value_467.append(i)
    # load in trajs, get binarized contacts
    dists_bin_all = generate_neighbor_binarized_contacts(traj_list, top_file, atom_indices, index_value_467, unique_resis)
    # save the binarized data
    np.save(f"{output_file1_name}",dists_bin_all)
    # take the average of the binarized data for each pair
    averaged_dists_bin = np.average(np.concatenate(dists_bin_all), axis=0)
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
    plt.ylabel(f"{set1_name}", fontsize=10)
    plt.xlabel(f"{set2_name} (residue number)", fontsize=10)
    plt.title(f"{matrix_name}")
    plt.tight_layout()
    plt.savefig(f"{matrix_name}")


# Wild type:
top_file = top_files[0]
path_to_traj_list = path_to_traj_lists[0]

set1 = [467]
set2 = [a for a in generate_neighbor_atom_ind(top_file, 467)[1] if a != 467]

set1_name = "467"
set2_name = "neighbors"

output_file1_name = f"{set1_name}_{set2_name}_binarized_distances_wildtype"
output_file2_name = f"{set1_name}_{set2_name}_averaged_binarized_distances_wildtype"
matrix_name = f"{set1_name}_{set2_name}_wildtype"
neighbors_wt(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name)


# mutant:
top_file_wt = top_files[0]
top_file = top_files[1]
path_to_traj_list = path_to_traj_lists[1]

set1 = [467]
set2 = [a for a in generate_neighbor_atom_ind(top_file_wt, 467)[1] if a != 467]

set1_name = "467"
set2_name = "neighbors"

output_file1_name = f"{set1_name}_{set2_name}_binarized_distances_I467T"
output_file2_name = f"{set1_name}_{set2_name}_averaged_binarized_distances_I467T"
matrix_name = f"{set1_name}_{set2_name}_I467T"
neighbors_mt(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name,top_file_wt)
