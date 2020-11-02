import pickle
import seaborn as sns
from matplotlib import pyplot as plt
with open('goid_human.pkl', 'rb') as f:
    # Pickle will store our object into the specified file
    ggid = pickle.load(f)
lister = [ 122,6357,7165,7186,45944]
vv = []
for o in lister:
    vv.append(ggid[o])
print(vv)
ax= sns.barplot(lister, vv)
plt.title("Annotations in each goid", fontsize=24)
plt.ylabel('Number of Annotations', fontsize=18)
plt.xlabel('GoID', fontsize=18)
#adding the text labels
print('2')
rects = ax.patches
print(rects)
labels = vv
print('3')
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=8)
plt.show()