from collections import defaultdict
from tqdm import tqdm
import pickle
with open("../../data/goa_human.gaf", "r", encoding='utf-8') as infile:
    n_items = 0
    for lines in infile:
        n_items += 1
    infile.seek(0)
    go_dict = defaultdict(set)
    for line in tqdm(infile, total=n_items):
        line = line.strip()
        if not line.startswith("!"):
            items = line.split("\t")
            if not items[4].startswith('GO'):
                print(items[4])
            key, value = items[1], int(items[4][3:])
            go_dict[key].add(value)
print('finished uniprot')
print("ordered dictionary done")
with open('uni2go_human.pkl', 'wb') as f:
    pickle.dump(go_dict, f)
    print('done')


