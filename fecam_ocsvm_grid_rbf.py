import itertools
import subprocess
import json

nu = 0.99
kernel = 'rbf'
gamma = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1, 3, 5, 8, 9, 9.5, 10, 10.5, 11, 12, 15, 17, 20, 30, 40, 50, 60
]

config_path = './exps/FeCAM_cifar100.json'

for g in gamma:
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['ocsvm_nu'] = nu
    data['ocsvm_gamma'] = g
    data['ocsvm_kernel'] = kernel
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
    
