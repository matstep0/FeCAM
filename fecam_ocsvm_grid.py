import itertools
import subprocess
import json

nu = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
gamma = [0.001, 0.01, 0.1, 1, 10, 100]
kernel = ['rbf', 'poly', 'sigmoid']

config_path = './exps/FeCAM_cifar100.json'

print(len(list(itertools.product(nu, gamma, kernel))))

for (n, g, k) in itertools.product(nu, gamma, kernel):
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['ocsvm_nu'] = n
    data['ocsvm_gamma'] = g
    data['ocsvm_kernel'] = k
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
    
