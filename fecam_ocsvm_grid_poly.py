import itertools
import subprocess
import json

nu = 0.3
gamma = 0.01
kernel = 'poly'
degree = [2, 3, 4, 5, 6]
coef0 = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

config_path = './exps/FeCAM_cifar100.json'

for (d, c) in itertools.product(degree, coef0):
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['ocsvm_nu'] = nu
    data['ocsvm_gamma'] = gamma
    data['ocsvm_kernel'] = kernel
    data['ocsvm_degree'] = d
    data['ocsvm_coef0'] = c
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
    
