import pickle
with open('/../preprocessing/goid_human.pkl', 'rb') as f:
    # Pickle will store our object into the specified file
    ggid = pickle.load(f)

print(ggid.values())