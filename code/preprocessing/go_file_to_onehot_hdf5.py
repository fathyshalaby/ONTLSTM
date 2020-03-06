# -*- coding: utf-8 -*-
"""recursive_dag_to_one_hot.py: Turn a dictionary representing a GO-DAG into a one-hot array


Author -- Michael Widrich
Created on -- 2018-12-26
Contact -- michael.widrich@jku.at
"""
import re
import numpy as np
import h5py
import mmap

go_filename = '../../data/go-basic.obo'

with h5py.File('one_hot_go.h5py', 'w-') as h5file:
    pass
#
# Read obo file, get GO term keys
#
obo_term_regex = "^\[Term\]\n(.*?)\n\n"
obo_term_regex = re.compile(bytes(obo_term_regex, encoding='UTF-8'), re.DOTALL | re.MULTILINE)

with open(go_filename, "r") as gf:
    gf_mmap = mmap.mmap(gf.fileno(), 0, access=mmap.ACCESS_READ)
    obo_terms = obo_term_regex.findall(gf_mmap)

go_number_length = 7
id_linestart = b"id: GO:"
is_a_regex = "^is_a: GO:(.{" + str(go_number_length) + "})"
is_a_regex = re.compile(bytes(is_a_regex, encoding='UTF-8'), re.DOTALL | re.MULTILINE)
alt_id_regex = "^alt_id: GO:(.{" + str(go_number_length) + "})"
alt_id_regex = re.compile(bytes(alt_id_regex, encoding='UTF-8'), re.DOTALL | re.MULTILINE)

ancestor_dict = dict()
alt_id_dict = dict()
for obo_term in obo_terms:
    obo_term_head = obo_term.split(b'\n', maxsplit=3)
    if obo_term_head[2] == b'namespace: biological_process':  # only use biological_process namespace
        ancestor_dict[int(obo_term_head[0][len(id_linestart):])] = [int(s) for s in
                                                                    is_a_regex.findall(obo_term_head[3])]
        alt_id_dict.update([(int(s), int(obo_term_head[0][len(id_linestart):]))
                            for s in alt_id_regex.findall(obo_term_head[3])]
                           + [(int(obo_term_head[0][len(id_linestart):]), int(obo_term_head[0][len(id_linestart):]))])
go_ids = np.array(list(ancestor_dict.keys()), dtype=np.int)
go_ids.sort()
n_go_ids = len(go_ids)
go_id_to_ind = dict([(go_id, ind) for ind, go_id in enumerate(go_ids)])
# And you have a one-hot array (I'll use a numpy array but a hdf5 array would behave the same):
one_hot_dag = np.zeros((n_go_ids, n_go_ids), dtype=np.bool)
print(ancestor_dict)

# We use the ancestor_dict to set the one_hot_dag but we need a recursive function to traverse the ancestor_dict:
def recursive_is_a(dag_key, upper_leaves=()):
    """Function for recursively setting all is_a connections of a go term in one_hot_dag to True"""
    # Add the current key to the upper leaves
    upper_leaves += (dag_key,)
    # Set the key entry for itself and all it's upper leaves to True
    one_hot_dag[[go_id_to_ind[go_id] for go_id in upper_leaves], go_id_to_ind[dag_key]] = True
    # And then use this function to set all is_a connections and their is_a connections to True recursively
    is_a_connections = ancestor_dict[dag_key]
    [recursive_is_a(is_a_keys, upper_leaves) for is_a_keys in is_a_connections]


def process_go_id(i):
    """Add some verbose output"""
    go_id = go_ids[i]
    recursive_is_a(go_id)
    print("processed go id {} of {}".format(i, len(go_ids)))

[process_go_id(i) for i in range(len(go_ids))]

go_id_to_ind_array = np.full((np.max(go_ids)+1,), fill_value=-1, dtype=np.int)
go_id_to_ind_array[go_ids] = np.arange(len(go_ids))

with h5py.File('one_hot_go.h5py', 'w') as h5file:
    h5file.create_dataset('one_hot_dag', data=one_hot_dag, compression="lzf", chunks=True)
    h5file.create_dataset('index_to_go_id', data=go_ids, compression="lzf", chunks=True)
    h5file.create_dataset('go_id_to_index', data=go_id_to_ind_array, compression="lzf", chunks=True)


#
# Print some statistics
#

print("Number of is_a connections for all go terms:\n{}".format(list(zip(go_ids, one_hot_dag.sum(axis=1)))))

print("Total is_a connections found: {}".format(one_hot_dag.sum()))

print("is_a relations of go_id {}: {}".format(go_ids[0], np.asarray(one_hot_dag[0], np.int)))

print("Average number of is_a relations per go_id {}".format(np.mean(one_hot_dag.sum(axis=1))))

#
# Using the array:
#

# Let's say we have 2 sequences. Sequence 0 is associated with GO id 2 and 3 and sequence 1 is associated with GO
# term 5. Then the input features for the neural network would look like this:
with h5py.File('one_hot_go.h5py', 'r') as h5file:
    index_to_go_id = h5file['index_to_go_id'][:]
    go_id_to_index = h5file['go_id_to_index'][:]
input_features = np.zeros((2, len(index_to_go_id)), dtype=np.float32)
sequence_dict = {0: np.array([2, 3]), 1: np.array([30702])}  # this dictionary represents the sequence to go-term relations

for sequence in sequence_dict.keys():
    for go_ids_in_sequence in sequence_dict[sequence]:
        one_hot_dag_indices = go_id_to_index[go_ids_in_sequence]
        if np.any(one_hot_dag_indices == -1):
            raise ValueError("Unknown GO ID {}".format(go_ids_in_sequence[one_hot_dag_indices == -1]))
        input_features[sequence, one_hot_dag[one_hot_dag_indices]] = 1.

# And now we have the 1-hot input features:
print(input_features)
print("Sequence 0 has go_id_indices ", np.where(input_features[0]))
print("Sequence 0 has go_ids ", index_to_go_id[np.where(input_features[0])[0]])
print("Sequence 1 has go_ids ", index_to_go_id[np.where(input_features[1])[0]])
