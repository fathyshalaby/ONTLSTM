import pickle
from collections import OrderedDict
import uniprotparser
import numpy as np
import Bio.SeqIO as SeqIO
forbidden = ['B', 'O', 'J', 'U', 'X', 'Z']
def sequence(inputfile):
    finaldic = dict()
    p = OrderedDict(uniprotparser.go_dict)
    for record in SeqIO.parse(inputfile, "fasta"):
        print(record.id.split('|')[1])
        if record.id.split('|')[1] in p.keys():
            if not any([x in record.seq for x in forbidden]):
                finaldic[record.id.split('|')[1]] = record.seq
            else:
                pass
    return finaldic
m = sequence('../../data/human_swissprot_sequences.fasta')
z = OrderedDict(m)
p = max(z.values())
with open('fullsequence_swiss_human.pkl', 'wb') as f:
    pickle.dump(m, f)
    print('done')
    print(z)
print(sequence('../../data/human_swissprot_sequences.fasta'))



'''
def sequence(inputfile):
    c = gzip.decompress(inputfile)
    file = open(c ,'r')
    fastadic = dict()
    finaldic = dict()
    blocks = file.read()
    block = blocks.split('>')
    for entries in block[1:]:
        code = entries.split('\n')[0].split(' ')[0][len('Uniref100_'):]
        sequences = entries.split('\n')[1:]
        sequence = ''.join(sequences)
        fastadic[code] = sequence
    uniprot = uniprotparser.uni8go
    for key in fastadic.keys():
        for id in uniprot.keys():
            if key == id:
                finaldic[id] = fastadic[key]
    return finaldic
m = sequence('uniref_100.fasta')
z = OrderedDict(m)
p = max(z.values())
with open('fullsequence.pkl', 'wb') as f:
    pickle.dump(m, f)
    print('done')
    print(z)
    '''