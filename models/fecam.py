
import logging
import numpy as np
from tqdm import tqdm
import torch

from torch import nn
from torch import optim
from torch import linalg as LA
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import CosineIncrementalNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from torchvision import datasets, transforms
from utils.autoaugment import CIFAR10Policy
from utils.maha_utils import compute_common_cov, compute_new_common_cov, compute_new_cov
from sklearn import svm
from collections import namedtuple
from sklearn.covariance import MinCovDet
import itertools
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

EPSILON = 1e-8


class FeCAM(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = CosineIncrementalNet(args, False)
        self._protos = []
        self._init_protos = []
        self._common_cov = None
        self._cov_mat = []
        self._diag_mat = []
        self._common_cov_shrink = None
        self._cov_mat_shrink = []
        self._norm_cov_mat = []
        self._ocsvm_models = []
        self._pca = []
        self._pca_protos = []
        self._pca_cov = []

    def after_task(self):
        self._known_classes = self._total_classes
        # if self._cur_task == 0:
        #     self.save_checkpoint("{}_{}_{}_{}".format(self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"]))
        
    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1
        if self.args['dataset'] == "cifar100":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63/255),
                CIFAR10Policy(),
                transforms.ToTensor(),
            ]
        elif self.args['dataset'] == "tinyimagenet200":
            self.data_manager._train_trsf = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]
        elif self.args['dataset'] == "imagenet100":
            self.data_manager._train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.ToPILImage()
            ]

        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task)
        self._network_module_ptr = self._network
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task > 0:   # Freezing the network
            for p in self._network.convnet.parameters():
                p.requires_grad = False
        
        self.shot = None

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', shot=self.shot)  
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        resume = True  # set resume=True to use saved checkpoints after first task
        if self._cur_task == 0:
            if resume:
                self._network.load_state_dict(torch.load("{}_{}_{}_{}_{}.pkl".format(self.args["dataset"],self.args["model_name"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            if not resume:
                self._epoch_num = self.args["init_epochs"]
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
                )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=self.args["init_epochs"])
                self._train_function(train_loader, test_loader, optimizer, scheduler)        
            self._build_base_protos()
            self._build_protos()

            self.train_pca(train_loader)

            if self.args["full_cov"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    if self.args["shrink"]:  # we apply covariance shrinkage 2 times to obtain better estimates of matrices
                        for cov in self._cov_mat:
                            self._cov_mat_shrink.append(self.shrink_cov(cov))
                    if self.args["norm_cov"]:
                        self._norm_cov_mat = self.normalize_cov()
                else:
                    self._common_cov = compute_common_cov(train_loader, self)
            elif self.args["diagonal"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    for cov in self._cov_mat:
                        self._cov_mat_shrink.append(self.shrink_cov(cov))
                    for cov in self._cov_mat_shrink:
                        cov = self.normalize_cov2(cov)
                        self._diag_mat.append(self.diagonalization(cov))
        else:
            self._cov_mat_shrink, self._norm_cov_mat, self._diag_mat = [], [], []
            self._network.to(self._device)
            if hasattr(self._network, "module"):
                self._network_module_ptr = self._network.module
            self._build_protos()
            self._update_fc()

            self.train_pca(train_loader)

            if self.args["full_cov"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    if self.args["shrink"]:
                        for cov in self._cov_mat:
                            self._cov_mat_shrink.append(self.shrink_cov(cov))
                    if self.args["norm_cov"]:
                        self._norm_cov_mat = self.normalize_cov()
                else:
                    self._common_cov = compute_new_common_cov(train_loader, self)
            elif self.args["diagonal"]:
                if self.args["per_class"]:
                    compute_new_cov(self)
                    for cov in self._cov_mat:
                        self._cov_mat_shrink.append(self.shrink_cov(cov))
                    for cov in self._cov_mat_shrink:
                        cov = self.normalize_cov2(cov)
                        self._diag_mat.append(self.diagonalization(cov))
                    
                    
    # PCA + (n1 | n2 | maha | ocsvm) ------------------

    def train_pca(self, train_loader):

        # From the results of grid search, for each kernel is rbf
        ocsvm_best_params_per_task = [
            { 'gamma': 0.01, 'nu': 0.7 }, # 0
            { 'gamma': 0.1, 'nu': 0.9 },  # 1
            { 'gamma': 0.1, 'nu': 0.9 },  # 2
            { 'gamma': 0.1, 'nu': 0.9 },  # 3
            { 'gamma': 0.1, 'nu': 0.9 },  # 4
            { 'gamma': 0.1, 'nu': 0.9 },  # 5
        ]
        
        vectors, y_true = self._extract_vectors(train_loader)

        if (self.args['pca_vecnorm']):
            print('Normalising the embedded train vectors before PCA')
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        # Organise the data
        classes = np.unique(y_true)
        class_to_data = {cls: [] for cls in classes}
        for vector, label in zip(vectors, y_true):
            class_to_data[label].append(vector)

        # Fit PCA, cov and One-class SVM
        for cls, data in sorted(class_to_data.items()):
            print('Processing class:', cls)

            if self.args["tukey"]:
                data = self._tukeys_transform(data)

            pca = PCA(n_components=self.args['pca_components'])
            pca_data = pca.fit_transform(data)
            self._pca.append(pca)

        # COV #1 - daje słabe wyniki, chyba poprawnie
            # cov = MinCovDet(random_state=0).fit(pca_data)
            # pca_cov = torch.tensor(cov.covariance_)  

        # COV #2 - daje 512 wymiarową kowariancję, źle
            # pca_cov = torch.from_numpy(pca.get_covariance())
            
        # COV #3 - dobre wyniki, porównywalne z FeCAM przy PROTO #2, chyba poprawnie
            pca_cov = torch.from_numpy(np.cov(pca_data, rowvar=False))

        # COV #4 - złe wyniki w połączeniu z PROTO #1, z PROTO #2 nie sprawdzone
            # pca_cov = torch.from_numpy(np.cov(pca.transform(vectors), rowvar=False)) 

            self._pca_cov.append(pca_cov)
            
        # PROTO #1 - bardzo słabe wyniki dla dowonego COV, chyba poprawnie
            # pca_proto = torch.tensor(np.mean(pca_data, axis=0)).to(self._device)       

        # PROTO #2 - dobre wyniki w połączeniu z COV #3, chyba niepoprawnie
            pca_proto = torch.tensor(np.mean(pca.transform(vectors), axis=0)).to(self._device)

            self._pca_protos.append(pca_proto)

            if self.args['pca_dist'] == 'ocsvm':
                print('Traning one-class SVM for class:', cls)
                best_params = ocsvm_best_params_per_task[self._cur_task]
                ocsvm = svm.OneClassSVM(gamma=best_params['gamma'], nu=best_params['nu'], kernel='rbf').fit(pca_data)
                self._ocsvm_models.append(ocsvm)

        # Shrink PCA covariance - two times
        for _ in range(2):
            for cls in sorted(class_to_data.keys()):
                self._pca_cov[cls] = self.shrink_cov(self._pca_cov[cls])

    # ------------------ PCA + (n1 | n2 | maha | ocsvm) 


    def _build_base_protos(self):
        for class_idx in range(self._known_classes, self._total_classes):
            class_mean = self._network.fc.weight.data[class_idx]
            self._init_protos.append(class_mean)

    def _build_protos(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', shot=self.shot, ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._protos.append(torch.tensor(class_mean).to(self._device))

    def _update_fc(self):
        self._network.fc.fc2.weight.data = torch.stack(self._protos[-self.args["increment"]:], dim=0).to(self._device)  # for cosine incremental fc layer
    
    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epoch_num))
        for _, epoch in enumerate(prog_bar):
            if self._cur_task == 0:
                self._network.train()
            else:
                self._network.eval()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                if self._cur_task ==0:
                    logits = self._network(inputs)['logits']
                else:
                    logits = self._network_module_ptr.fc(inputs)['logits']
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)
