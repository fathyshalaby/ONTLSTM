 # -*- coding: utf-8 -*-
"""preprocessing.py: tools for data pre-processing


Author -- Michael Widrich
Contact -- widrich@ml.jku.at
"""

import numpy as np
from torch.utils.data import Dataset


class TensorDataset(Dataset):

    def __init__(self, annotation, sequences,gid,ggid):
        self.ANNOTATION = list(annotation.keys())
        self.SEQUENCE = sequences
        self.ANNOTATION = [a for a in self.ANNOTATION if a in self.SEQUENCE.keys()]
        self.ANNOTATIONs = annotation
        self.goid = gid
        self.gomatrix = gid['one_hot_dag'][:]
        self.goid_to_index = gid['go_id_to_index'][:]
        self.vocab = 'MAFSEDVLKYRPNWQCGIHT'
        self.aa_lookup = dict([(k, v) for v, k in enumerate(self.vocab)])
        self.ggid=np.array(list(ggid.values()))
        self.allowed_goids = np.isin(list(ggid.keys()),[45944])
        self.n_allowed_goids = self.allowed_goids.sum()
        self.n_features = len(self.vocab)
        self.n_classes = self.n_allowed_goids

    def __len__(self):
        return len(self.ANNOTATION)

    def __getitem__(self, idx):
        p = []
        uniprodid = self.ANNOTATION[idx]
        go_matrix= self.gomatrix
        list_of_go_ids = self.ANNOTATIONs[uniprodid]
        label = np.zeros_like(go_matrix[0], dtype=np.float32).flatten()
        for go_id in list_of_go_ids:
            goid_to_idx = self.goid_to_index[go_id]
            if goid_to_idx >= 0:
                label += go_matrix[goid_to_idx]
        label[:] = label > 0
        label = label[self.allowed_goids]#its always 12326 different labels which is false
        #print(len(list_of_go_ids),len(label))
        sequence = self.SEQUENCE[uniprodid]#works but give key error will be solved when do it on full files
        x = np.zeros(shape=(len(sequence), len(self.aa_lookup)), dtype=np.float32)
        x[np.arange(len(sequence)), [self.aa_lookup[aa] for aa in sequence]] = 1
        sample = {'GoID': label, 'UniprotID': uniprodid, 'Sequence': x}
        return x, label, uniprodid
'''
    def __init__(self, annotation, sequences,gid,ggid):
        self.ANNOTATION = list(annotation.keys())
        self.SEQUENCE = sequences
        self.ANNOTATION = [a for a in self.ANNOTATION if a in self.SEQUENCE.keys()]
        self.ANNOTATIONs = annotation
        self.goid = gid
        self.gomatrix = gid['one_hot_dag'][:]
        self.goid_to_index = gid['go_id_to_index'][:]
        self.vocab = 'MAFSEDVLKYRPNWQCGIHTXZBUO'
        self.aa_lookup = dict([(k, v) for v, k in enumerate(self.vocab)])
        self.allowed_goids = np.array(list(ggid.values()))
        self.n_allowed_goids = self.allowed_goids.sum()
        self.nofclasses = self.n_allowed_goids
        self.n_features = len(self.vocab)
        self.n_classes = 30821


    def __len__(self):
        return len(self.ANNOTATION)

    def __getitem__(self, idx):
        uniprodid = self.ANNOTATION[idx]
        go_matrix= self.gomatrix
        list_of_go_ids = self.ANNOTATIONs[uniprodid]
        label = np.zeros_like(go_matrix[0], dtype=np.float32).flatten()
        for go_id in list_of_go_ids:
            goid_to_idx = self.goid_to_index[go_id]
            if goid_to_idx >= 0:
                label += go_matrix[goid_to_idx]
        label[:] = label > 0
        label = label[self.allowed_goids]
        sequence = self.SEQUENCE[uniprodid]#works but give key error will be solved when do it on full files
        x = np.zeros(shape=(len(sequence), len(self.aa_lookup)), dtype=np.float32)
        x[np.arange(len(sequence)), list(sequence)] = 1
        return x,label,uniprodid
        '''
'''class TensorDataset_(Dataset):
    def __init__(self,dataset):
        self.dataset = set(self.dataset.values())
        self.goids = np.array(self.dataset.keys())
        self.sequences = dataset.
        self.vocab = 'MAFSEDVLKYRPNWQCGIHTXZBUO'
        self.aa_lookup = dict([(k, v) for v, k in enumerate(self.vocab)])
        self.n_features = 25
        self.n_classes = len(self.goids)

    def __len__(self):
        return len(set(self.dataset.values()))

    def __getitem__(self, idx):
        label = nn.functional.one_hot
        sequence =self.sequence[idx]
        x = np.zeros(shape=(len(sequence), len(self.aa_lookup)), dtype=np.float32)
        x[np.arange(len(sequence)), [self.aa_lookup[aa] for aa in sequence]] = 1
        return label,x'''

