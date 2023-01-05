import mdtraj as md
import numpy as np
import itertools
import matplotlib.pyplot as plt

# the following lists should be set up like: (wildtype, I467T)
top_files = ["/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories_wt/myh7-5n6a-holo-prot-masses.pdb",
             "/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories/myh7-5n6a-i467t-adp-phos-prot-masses.pdb"]
path_to_traj_lists = ["/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories_wt/traj_list.txt",
                      "/Users/ssolieva/Desktop/bowman_lab/MSM_I467T/trajectories/traj_list.txt"]

def generate_neighbor_atom_ind(top_file):
    '''this will take a top_file + one residue and output neighboring atom indices for that residues.
        resis is a list of residue numbers'''
    topology = md.load(top_file)
    query_indices = topology.top.select('residue 467')
    haystack_indices = topology.top.select('protein')
    cutoff = 0.5
    atom_indices_ = md.compute_neighbors(topology,cutoff, query_indices, haystack_indices)
    resis =[]
    for i in atom_indices_[0]:
        #if int(str(topology.top.atom(i).residue)[-3:]) != 467:
            #print(int(str(topology.top.atom(i).residue)[-3:]))
        resis.append(int(str(topology.top.atom(i).residue)[-3:]))
    unique_resis = np.unique(resis)
    selection_string = ' or '.join(f'residue {r}' for r in unique_resis)
    atom_indices = topology.top.select(selection_string)
    return atom_indices,unique_resis

def generate_neighbor_binarized_contacts(traj_list, top_file, atom_indices, cutoff):
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
        pairs_ind = list(itertools.product(set1_ind, set2_ind))
        #print(pairs_ind)
        dists, pair_index = md.compute_contacts(trj,pairs_ind,scheme='closest-heavy')
        # binarize first, then save
        #print(pair_index)
        dists_bin_all.append((dists < cutoff).astype(np.bool_)) # cutoff in nm
    return dists_bin_all

def neighbors_wt(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name):
    # import traj_list
    with open(path_to_traj_list, "r") as fd:
        traj_list = fd.read().splitlines()
    # get atom indices
    atom_indices,unique_resis= generate_neighbor_atom_ind(top_file)
    index_value_467 = []
    for i in range(len(unique_resis)):
        if unique_resis[i] == 467:
            print("index value:",i)
            index_value_467.append(i)
    # load in trajs, get binarized contacts
    dists_bin_all = generate_neighbor_binarized_contacts(traj_list, top_file, atom_indices, cutoff=0.4)
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
    plt.ylabel(f"{set1_name} (residue number)", fontsize=10)
    plt.xlabel(f"{set2_name} (residue number)", fontsize=10)
    plt.title(f"{matrix_name}")
    plt.savefig(f"{matrix_name}")
    
def neighbors_mt(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name, top_file_wt):
    # import traj_list
    with open(path_to_traj_list, "r") as fd:
        traj_list = fd.read().splitlines()
    # get atom indices
    atom_indices,unique_resis= generate_neighbor_atom_ind(top_file_wt)
    index_value_467 = []
    for i in range(len(unique_resis)):
        if unique_resis[i] == 467:
            print("index value:",i)
            index_value_467.append(i)
    # load in trajs, get binarized contacts
    dists_bin_all = generate_neighbor_binarized_contacts(traj_list, top_file, atom_indices, cutoff=0.4)
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
    plt.ylabel(f"{set1_name} (residue number)", fontsize=10)
    plt.xlabel(f"{set2_name} (residue number)", fontsize=10)
    plt.title(f"{matrix_name}")
    plt.savefig(f"{matrix_name}")

# Wild type:
top_file = top_files[0]
path_to_traj_list = path_to_traj_lists[0]

set1 = unique_resis[index_value_467]
set2 = unique_resis
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

set1 = unique_resis[index_value_467]
set2 = unique_resis
set1_name = "467"
set2_name = "neighbors"

output_file1_name = f"{set1_name}_{set2_name}_binarized_distances_I467T"
output_file2_name = f"{set1_name}_{set2_name}_averaged_binarized_distances_I467T"
matrix_name = f"{set1_name}_{set2_name}_I467T"
neighbors_mt(set1, set2, set1_name, set2_name, top_file, path_to_traj_list, output_file1_name, output_file2_name, matrix_name,top_file_wt)


