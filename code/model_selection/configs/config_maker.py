import random
import json
import subprocess
import glob
dic = {"device": "cuda:2",
       "num_threads": 11,
       "lr": 1e-2,
       "batch_size": 32,
       "n_lstm": 8,
       "n_updates": 4e4,
       "results_dir": "results_pre",
       "print_stats_at": 1e2,
       "plot_at": 1e3,
       "validate_at": 5e2,
       "n_samples": 1e5,
       "rnd_seed": 42,
	    "opt" :'adam',
        "classification":'binary',
       "L2":1e-5,
       "kernel_size": "3",
       "use_cnn":True,
       "use_prefinal_layer":True
       }
BATCH_SIZE = [8, 16, 32, 64, 128, 256]
LEARNING_RATE = [1,1e-1, 1e-2, 1e-3,1e-4]
LSTM_BLOCKS = [2,4,8, 16, 32]
use = [True,False]
kernel = [3,5]
optim = ['sgd','adam']
decay = [1e-6,1e-5,1e-4,1e-3,1e-2]
n_configs = len(BATCH_SIZE) * len(LEARNING_RATE) * len(LSTM_BLOCKS)
print(n_configs, 'number of configurations')
for i in range(n_configs):
    dic["results_dir"] = 'results'+str(i)
    dic["batch_size"] = random.choice(BATCH_SIZE)
    dic["lr"] = random.choice(LEARNING_RATE)
    dic["n_lstm"] = random.choice(LSTM_BLOCKS)
    dic["opt"] = random.choice(optim)
    dic["L2"] = random.choice(decay)
    dic["kernel_size"] = random.choice([3, 5])
    dic["use_cnn"] = random.choice(use)
    dic["use_prefinal_layer"] = random.choice(use)
    filename = 'config' + str(i) + '.json'
    file = open(filename, 'w')
    json.dump(dic, file)
    file.close()
    subprocess.call(['python3', '../main.py', filename])

