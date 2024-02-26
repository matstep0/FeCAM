import torch
import os
import pickle
import numpy as np
import torch.nn as nn
from torch import linalg as LA
from torch.utils.data import DataLoader


def compute_common_cov(train_loader, model):
    if model.args["dataset"] == 'imagenet100':
        cov = model._extract_vectors_common_cov(train_loader)
    else:
        vectors, _ = model._extract_vectors(train_loader)
        if model.args["tukey"]:
            vectors = model._tukeys_transform(vectors)
        cov = torch.tensor(np.cov(vectors.T))
    return cov

def compute_new_common_cov(train_loader, model):
    cov = compute_common_cov(train_loader, model)
    if model.args["shrink"]:
        cov = model.shrink_cov(cov)
    ratio = (model._known_classes/model._total_classes)

    common_cov = ratio*model._common_cov + (1-ratio)*cov
    return common_cov


# Dla taska = 0 brana jest macierz jednostkowa 30x30 zamiast kowariancji - wtedy wyniki są dobre
# Dla kolejnych tasków wyniki są fatalne ( FeCAM top1 curve: [83.56, 12.22, 8.89, 10.1, 7.34, 8.48] )

# Wyłączanie, włączanie shrink - brak poprawy
# Wyłączanie, włączanie pca_vecnorm - brak poprawy
# Sprawdzenie czy pca.transform(vectors.T) da poprawę - nie dało
# Tukey - jest wyłączony

def compute_new_cov(model):
    for class_idx in range(model._known_classes, model._total_classes):
        data, targets, idx_dataset = model.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                mode='test', ret_data=True)
        idx_loader = DataLoader(idx_dataset, batch_size=model.args["batch_size"], shuffle=False, num_workers=4)
        vectors, _ = model._extract_vectors(idx_loader)

        # PCA + (n1 | n2 | maha | ocsvm) ------------------

        pca = model._pca[class_idx]
        vectors = pca.transform(vectors)

        # ------------------ PCA + (n1 | n2 | maha | ocsvm) 

        if model.args["tukey"]:
            vectors = model._tukeys_transform(vectors)
        
        cov = torch.tensor(np.cov(vectors.T))
        
        # First shrink
        if model.args["shrink"]:
            cov = model.shrink_cov(cov)

        model._cov_mat.append(cov)
