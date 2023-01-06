# the following lists should be set up like: (wildtype, I467T)
top_files = ["/home/artur/bowmore-backup/i467t/wt-trajectories/myh7-5n6a-holo-prot-masses.pdb",
             "/home/artur/bowmore-backup/i467t/i467t-trajectories/myh7-5n6a-i467t-adp-phos-prot-masses.pdb"]
path_to_traj_lists = ["wt-trajectories.txt",
                      "i467t-trajectories.txt"]

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

def generate_neighbor_atom_ind(top_file_wt, top_file, query_residue, cutoff=0.5):
    '''this will output neighboring atom indices for a specified residue (query_residue).
        top_file_wt is needed to make sure the same resis list is being used in the mutant.
            If finding neighbors for the wt system, then top_file_wt == top_file.
        top_file is needed to make sure the correct atom indices are being used (wt indices should not be used for mutant).'''
    # make a list of neighboring residues (this part uses top_file_wt)
    wt_topology = md.load(top_file_wt)
    query_indices = wt_topology.top.select(f'residue {query_residue}')
    haystack_indices = wt_topology.top.select('all')
    atom_indices = md.compute_neighbors(wt_topology,cutoff, query_indices, haystack_indices)[0]
    unique_resis = np.unique(
        [wt_topology.top.atom(ai).residue.resSeq for ai in atom_indices]
    )
    unique_resis = [r for r in unique_resis if np.abs(r - query_residue) > 1] # remove neighbors
    unique_resis.append(query_residue) # add the query residue back in
    unique_resis = np.sort(unique_resis)
    selection_string = ' or '.join(f'residue {r}' for r in unique_resis)
    # make a list of neighboring atom indices (this part uses top_file)
    topology = md.load(top_file) # load in top_file that you want indices for
    atom_indices = topology.top.select(selection_string)
    return atom_indices, unique_resis

def generate_binarized_contacts(traj_list, top_file, atom_indices, cutoff=0.4):
    '''this will load trajs, compute closest heavy atom distances,
    binarize the distances, and save the binarized distances '''
    dists_bin_all = []
    for i in traj_list:
        trj = md.load(i, top=top_file, atom_indices=atom_indices) # load in trajectory with only certain residues
        print(i, trj)
        set1_ind = [] # use index values instead of actual residue numbers
        for i in set1:
            set1_ind.append(list(np.unique(np.concatenate([set1,set2]))).index(i))
        set2_ind = []
        for i in set2: # use index values instead of actual residue numbers
            set2_ind.append(list(np.unique(np.concatenate([set1,set2]))).index(i))
        pairs_ind = list(itertools.product(set1_ind, set2_ind)) # (a,a) is possible here
        dists, pair_index = md.compute_contacts(trj,pairs_ind,scheme='closest-heavy')
        dists_bin_all.append((dists < cutoff).astype(np.bool_)) # binarize first, then save
    return dists_bin_all

def any_pairs(set1, set2, set_name, top_file, top_file_wt, path_to_traj_list, system, query_residue=False, neighbors=False):
    ''' If neighbors = True, then set2 needs to be
            set2 = [a for a in generate_neighbor_atom_ind(top_file_wt, top_file, query_residue)[1] if a != query_residue]
            Also, need to specify a query residue.
        If neighbors = False, then you have to provide a set2 (list of residues)'''
    with open(path_to_traj_list, "r") as fd: # import traj_list
        traj_list = fd.read().splitlines()
    pairs_res = np.concatenate([set1, set2]) # combined list
    if neighbors == False:
        atom_indices = generate_atom_ind(top_file, resis = pairs_res) # get atom indices
    if neighbors == True:
        assert len(set1) == 1, 'set 1 is more than 1 residue, we should only be computing neighbors'
        atom_indices = generate_neighbor_atom_ind(top_file_wt, top_file, query_residue)[0] # get atom indices
    dists_bin_all = generate_binarized_contacts(traj_list, top_file, atom_indices) # load in trajs, get binarized contacts
    np.save(f"{set_name[0]}_{set_name[1]}_binarized_distances_{system}",dists_bin_all)  # save the binarized data
    averaged_dists_bin = np.average(np.concatenate(dists_bin_all), axis=0) # take the average of the binarized data for each pair
    np.save(f"{set_name[0]}_{set_name[1]}_averaged_binarized_distances_{system}",averaged_dists_bin) # save the averaged data
    shape = ( len(set1), len(set2) ) # reshape data into matrix form and then graph the matrix
    reshaped_dists = averaged_dists_bin.reshape( shape )
    fig, ax = plt.subplots()
    ax.matshow(reshaped_dists, vmin=0, vmax=1) # same colorbar
    for (i, j), z in np.ndenumerate(reshaped_dists):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',fontsize=8)
    plt.xticks(np.arange(len(set2)), set2)
    plt.yticks(np.arange(len(set1)), set1)
    plt.ylabel(f"{set_name[0]}", fontsize=14)
    plt.xlabel(f"{set_name[1]}", fontsize=14)
    plt.title(f"{set_name[0]}_{set_name[1]}_{system}")
    plt.tight_layout()
    plt.savefig(f"{set_name[0]}_{set_name[1]}_{system}")

# this applies to both systems:
top_file_wt = top_files[0]

# WILDTYPE:
top_file = top_files[0]
path_to_traj_list = path_to_traj_lists[0]
system = 'wildtype'
#  # S2 - relay helix
#  set_name = ["S2", "relay_helix"] #[set1, set2]
#  set1 = np.arange(463,473)
#  set2 = np.arange(473,490)
#  any_pairs(set1, set2, set_name, top_file, top_file_wt, path_to_traj_list, system, query_residue=False, neighbors=False)
#  # S1 - S2
#  set_name = ["S1", "S2"] #[set1, set2]
#  set1 = np.arange(232,245)
#  set2 = np.arange(463,473)
#  any_pairs(set1, set2, set_name, top_file, top_file_wt, path_to_traj_list, system, query_residue=False, neighbors=False)
#  # query_residue - neighbors
query_residue = 467
set_name = [f"{query_residue}", "neighbors"] #[set1, set2]
set1 = [query_residue]
set2 = [a for a in generate_neighbor_atom_ind(top_file_wt, top_file, query_residue)[1] if a != query_residue]
any_pairs(set1, set2, set_name, top_file, top_file_wt, path_to_traj_list, system, query_residue, neighbors=True)

query_residue = 787
set_name = [f"{query_residue}", "neighbors"] #[set1, set2]
set1 = [query_residue]
set2 = [a for a in generate_neighbor_atom_ind(top_file_wt, top_file, query_residue)[1] if a != query_residue]
any_pairs(set1, set2, set_name, top_file, top_file_wt, path_to_traj_list, system, query_residue, neighbors=True)

# MUTANT:
top_file = top_files[1]
path_to_traj_list = path_to_traj_lists[1]
system = 'I467T'
#  # S2 - relay helix
#  set_name = ["S2", "relay_helix"] #[set1, set2]
#  set1 = np.arange(463,473)
#  set2 = np.arange(473,490)
#  any_pairs(set1, set2, set_name, top_file, top_file_wt, path_to_traj_list, system, query_residue=False, neighbors=False)
#  # S1 - S2
#  set_name = ["S1", "S2"] #[set1, set2]
#  set1 = np.arange(232,245)
#  set2 = np.arange(463,473)
#  any_pairs(set1, set2, set_name, top_file, top_file_wt, path_to_traj_list, system, query_residue=False, neighbors=False)
#  # query_residue - neighbors
query_residue = 467
set_name = [f"{query_residue}", "neighbors"] #[set1, set2]
set1 = [query_residue]
set2 = [a for a in generate_neighbor_atom_ind(top_file_wt, top_file, query_residue)[1] if a != query_residue]
any_pairs(set1, set2, set_name, top_file, top_file_wt, path_to_traj_list, system, query_residue, neighbors=True)

query_residue = 787 # phosphate
set_name = [f"{query_residue}", "neighbors"] #[set1, set2]
set1 = [query_residue]
set2 = [a for a in generate_neighbor_atom_ind(top_file_wt, top_file, query_residue)[1] if a != query_residue]
any_pairs(set1, set2, set_name, top_file, top_file_wt, path_to_traj_list, system, query_residue, neighbors=True)
