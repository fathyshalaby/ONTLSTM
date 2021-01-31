import pickle
def readbin(src):
    with open(src,'rb') as f:
        output = pickle.load(f)
    return output