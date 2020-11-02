import pickle
with open('goid_human.pkl', 'rb') as f:
    # Pickle will store our object into the specified file
    ggid = pickle.load(f)


for key in ggid.keys():
    if ggid[key]>800:
        print(key,ggid[key])

