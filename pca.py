import subprocess
import json

# FeCAM settings: full cov matrix with shrink per class

# pca_vecnorm - embedding normalisation before PCA
# pca_components - num of components for PCA
# pca_dist - metric for classification
runs = [
    { 'pca_dist': 'maha',  'pca_vecnorm': True,  'pca_components': 30 },
    # { 'pca_dist': 'maha',  'pca_vecnorm': False, 'pca_components': 30 },
    # { 'pca_dist': 'norm1', 'pca_vecnorm': True,  'pca_components': 30 },
    # { 'pca_dist': 'norm1', 'pca_vecnorm': False, 'pca_components': 30 },
    # { 'pca_dist': 'norm2', 'pca_vecnorm': True,  'pca_components': 30 },
    # { 'pca_dist': 'norm2', 'pca_vecnorm': False, 'pca_components': 30 },
    # { 'pca_dist': 'ocsvm', 'pca_vecnorm': True,  'pca_components': 30 },
    # { 'pca_dist': 'ocsvm', 'pca_vecnorm': False, 'pca_components': 30 }
]

config_path = './exps/FeCAM_cifar100.json'

for run_params in runs:
    with open(config_path, 'r') as file:
        data = json.load(file)
    for param_name, param_value in run_params.items():
        data[param_name] = param_value
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f'Running FeCAM with params {run_params}')
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
    