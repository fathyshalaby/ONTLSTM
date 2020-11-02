import goparser
import pickle
from collections import OrderedDict
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import numpy as np
with open('fullsequence_swiss_human.pkl','rb') as f:
    data1 = pickle.load(f)
with open('uni2go_human.pkl','rb') as i:
    dat2 = pickle.load(i)
print(len(data1.keys()))
u,v = goparser.parseobofile('../../data/go-basic.obo')
k = []
an = OrderedDict()
print('extend')
for keys in tqdm(data1.keys()):
    k.extend(list(dat2[keys]))
print()
print('counting')
print()
for goid in tqdm(u.keys()):
    if goid in an.keys():
        pass
    elif k.count(goid):
        an[goid] = k.count(goid)
    else:
        an[goid] = 0
print(len(an.keys()))
with open('goid_human.pkl', 'wb') as f:
    pickle.dump(an, f)
    print('done')
print(len(an.keys()))
plt.plot(list(an.keys()),list(an.values()))
print(np.max(list(an.values())))
plt.show()



from collections import Counter
print(Counter(list(an.values())))
print(an)#on average we need 43 samples for each go id that is valid
print('creating a balanced dataset')
minimum_samples = 200
maximum_samples = 200
relationship_as_is = goparser.rel
balanceddataset ={}
goid_seq={}
for key in tqdm(data1.keys()):
    for value in dat2[key]:
        if value in goid_seq.keys():
            if type(data1[key]) is list:
                print(data1[key])
            goid_seq[value].append(data1[key])
        else:
            goid_seq[value] = [data1[key]]
print(len(goid_seq.keys()))
#check the number of connected sequences and get the go ids that are in the specified range
copo = 0
for goids in tqdm(goid_seq.keys()):
    copo+=1
    if minimum_samples<len(goid_seq[goids])<maximum_samples:
        print('inrange',goids)
        balanceddataset[goids] = goid_seq[goids]
    else:
        while len(goid_seq[goids]) > maximum_samples: #if the goid has a higher amount than the minimum_samples we get rid of the extra sequences which are also found in other goids/relatives(look at the relationship which is done in the goparser script)
            if goids in relationship_as_is.keys():
                for sequenceo in goid_seq[goids][200:]:
                    for relat in relationship_as_is[goids]:
                        if relat in goid_seq.keys():
                            print('adding',relat,sequenceo)
                            goid_seq[relat].append(sequenceo)
                print(len(goid_seq[goids]))
                goid_seq[goids]=goid_seq[goids][:200]
            else:
                print(len(goid_seq[goids]))
                goid_seq[goids] = goid_seq[goids][:200]
                print('you have been 200it', '|iteration',copo,'from',len(goid_seq.keys()))

        while len(goid_seq[goids])<minimum_samples:
            print('copying last sequence',len(goid_seq[goids]))
            goid_seq[goids].append(goid_seq[goids][-1])
balanceddataset=OrderedDict(goid_seq)
from matplotlib import pyplot as plt
print('creating plot')
print(len(balanceddataset.keys()))
seqnum = {}
for key in tqdm(u.keys()):
    if key in balanceddataset.keys():
        balanceddataset[key]=balanceddataset[key][:200]
        seqnum[key]= len(balanceddataset[key])
    else:
        seqnum[key]=0
print(len(seqnum))
print(sorted(list(seqnum.values()))[0],sorted(list(seqnum.values()))[-1],np.mean(sorted(list(seqnum.values()))))
plt.plot(list(seqnum.keys()),list(seqnum.values()))
plt.show()


with open('fullsequence_balanced.pkl', 'wb') as f:#{goid:sequences}
    pickle.dump(balanceddataset, f)
    print('done')
with open('goid_balanced.pkl', 'wb') as f:
    pickle.dump(seqnum, f)
    print('done')

newuni2go = {}
seq_uni={v: k for k, v in data1.items()}
print(seq_uni)
for keys in tqdm(balanceddataset.keys(),desc='goids'):#all goids:seq(many)[]
    for val in balanceddataset[keys]:
        if seq_uni[val] in newuni2go.keys():
            newuni2go[seq_uni[val]].append(keys)
        else:
            newuni2go[seq_uni[val]]=[keys]

print(list(OrderedDict(newuni2go).values())[1])
m = OrderedDict(newuni2go)
with open('uni2go_balanced.pkl', 'wb') as f:
    pickle.dump(m, f)
    print('done')



