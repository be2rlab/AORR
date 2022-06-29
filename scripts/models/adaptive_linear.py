# Copyright (C) 2022 ITMO University
# 
# This file is part of Adaptive Object Recognition For Robotics.
# 
# Adaptive Object Recognition For Robotics is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Adaptive Object Recognition For Robotics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Adaptive Object Recognition For Robotics.  If not, see <http://www.gnu.org/licenses/>.

import torch
from sklearn.mixture import GaussianMixture
from pytorch_metric_learning import losses, miners
from tqdm.auto import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import time


class AdLinear(torch.nn.Module):
    def __init__(self, in_dims, init_out_dims=1):
        super().__init__()

        self.prev_rows = None
        self.cur_rows = torch.nn.Linear(in_dims, init_out_dims)

    def forward(self, x):

        out = self.cur_rows(x)
        if self.prev_rows:
            out = torch.cat((self.prev_rows(x), out), dim=1)
        return out

    def expand(self, n_new_rows=1, freeze_prev=True):
        # merge last and current layers
        cur_weight = self.cur_rows.weight
        cur_bias = self.cur_rows.bias
        if self.prev_rows:
            prev_weight = self.prev_rows.weight
            prev_bias = self.prev_rows.bias

            cur_weight = torch.cat((prev_weight, cur_weight))
            cur_bias = torch.cat((prev_bias, cur_bias))

        device = self.cur_rows.weight.device

        self.prev_rows = torch.nn.Linear(*cur_weight.shape[::-1])
        self.prev_rows.to(device)

        self.prev_rows.weight = torch.nn.Parameter(data=cur_weight)
        self.prev_rows.bias = torch.nn.Parameter(data=cur_bias)

        self.cur_rows = torch.nn.Linear(cur_weight.shape[1], n_new_rows)

        self.prev_rows.to(device)
        self.cur_rows.to(device)

        if freeze_prev:
            self.prev_rows.weight.requires_grad = False

class DataSaver:
    def __init__(self) -> None:

        self.x = None
        self.y = None

    def save_prev_data(self, x, y):
        if self.x is not None:
            self.x = torch.cat((self.x, x))
            self.y = torch.cat((self.y, y))
        else:
            self.x = x
            self.y = y

    def get_data(self):
        return self.x, self.y

class GaussianMixtureModel:
    def __init__(self, n_components=5, covariance_type='full') -> None:
        self.gaussians = {}
        self.trained = False
        self.gaussian_mixture_params = {
            'n_components': n_components,
            'covariance_type': covariance_type
        }

    def add_classes(self, points, classes):

        assert len(points) == len(classes)

        for cl in torch.unique(classes):
            mixture_model = GaussianMixture(**self.gaussian_mixture_params)
            mixture_model.fit(points[classes == cl])
            self.gaussians[cl] = mixture_model

        self.trained = True

    def get_samples(self, n_samples, classes=None):

        if classes is None:
            classes = self.gaussians.keys()

        generated_points = torch.zeros(
            (n_samples * len(classes), self.gaussians[list(classes)[0]].n_features_in_), dtype=torch.float32)
        generated_labels = torch.zeros((n_samples * len(classes)))

        for idx, cl in enumerate(classes):
            cur_points = self.gaussians[cl].sample(n_samples)[0]
            generated_points[idx *
                             n_samples:(idx + 1) * n_samples] = torch.Tensor(cur_points)
            generated_labels[idx * n_samples:(idx + 1) * n_samples] = cl

        return generated_points, generated_labels


class AdaptiveLayerWrapper:
    
    def __init__(self, in_dim ,init_out_space_dim, mixture_n_prev_points_per_class=50, 
                counter_thresh=3,
                lr=1,
                weight_decay=0.0,
                batch_size=32,
                score_diff_thresh=0.05,
                loss_patience=10,
                epochs=30) -> None:

        self.layer = AdLinear(in_dim, init_out_dims=init_out_space_dim)
        self.mixture = GaussianMixtureModel(n_components=10,
                                            covariance_type='diag')
        self.val_saver = DataSaver()

        self.mixture_n_prev_points_per_class = mixture_n_prev_points_per_class
        self.counter_thresh = counter_thresh
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.score_diff_thresh = score_diff_thresh
        self.epochs = epochs
        self.loss_patience = loss_patience

    def __call__(self, x):
        return self.layer(x)

    def train(self, cur_train_features, cur_train_labels):

        # generate previous data and concatenate current data with previous
        if self.mixture.trained:
            generated_features, generated_labels = self.mixture.get_samples(
                self.mixture_n_prev_points_per_class)
            train_features = torch.cat(
                (generated_features, cur_train_features))
            train_labels = torch.cat((generated_labels, cur_train_labels))      
        else:
            train_features = cur_train_features
            train_labels = cur_train_labels
        
        self.mixture.add_classes(cur_train_features, cur_train_labels)

        weights = make_weights_for_balanced_classes(train_labels)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, len(weights))

        train_dataset = torch.utils.data.TensorDataset(
            train_features, train_labels)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=0, sampler=sampler)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # get accuracy without adaptive fs
        generated_val_features, generated_val_labels = self.mixture.get_samples(
                self.mixture_n_prev_points_per_class // 5) # taker 20 % of each training data
        fe_accs = evaluate(train_features, train_labels,
                                        generated_val_features, generated_val_labels, per_class_metric=True)


        accs = evaluate(train_features, train_labels,
                                        generated_val_features, generated_val_labels, 
                                        model=self.layer,
                                        per_class_metric=True)
        if torch.mean(fe_accs) - torch.mean(accs) < self.score_diff_thresh:
            return

        counter = 0

        while counter < self.counter_thresh:
            # prepare optimzier and loss

            criterion = losses.TripletMarginLoss(margin=0.25)
            optimizer = torch.optim.Adam(
            self.layer.parameters(),
                self.lr, # linear scaling rule
                weight_decay=self.weight_decay, 
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.85)


            # run train loop

            train_model(self.layer, train_dataloader, optimizer, criterion, scheduler, device, self.epochs, loss_patience=self.loss_patience)
            # evaluate 


            accs = evaluate(train_features, train_labels,
                                            generated_val_features, generated_val_labels, 
                                            model=self.layer,
                                            per_class_metric=True)
            print(
                f'After training mean accuracy: {torch.mean(accs):.2f}, min: {torch.min(accs):.2f} for class {torch.argmin(accs)}/{len(accs)-1}')
            # expand if needed
            if torch.mean(fe_accs) - torch.mean(accs) > self.score_diff_thresh:
                print('Expanding layer')
                self.layer.expand(freeze_prev=False)
            else:
                break

            counter += 1




def train_model(model, train_dataloader, optimizer, criterion, scheduler, device, epochs, loss_patience=10):
    pbar = tqdm(total=len(train_dataloader))

    miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="all")

    accs = 0

    per_epoch_losses = []
    for i in range(epochs):
        pbar.reset()
        total_loss = 0

        # train stage
        model.train()
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device).long()
            optimizer.zero_grad()
            features = model(x)

            hard_pairs = miner(features, y)

            loss = criterion(features, y, hard_pairs)
            total_loss += loss.item()

            pbar.update()
            loss.backward()
            optimizer.step()

        scheduler.step()

        per_epoch_losses.append(total_loss)

        
        if (per_epoch_losses[-1] != min(per_epoch_losses)) and (i - torch.argmin(torch.Tensor(per_epoch_losses)) >= loss_patience):
            pbar.display()
            pbar.close()
            print(
                f'model did not improve since epoch {torch.argmin(torch.Tensor(per_epoch_losses))}, pruning')
            break


        pbar.set_description(
            f'epoch: {i+1}/{epochs}, Loss: {total_loss:.5f}, val acc: {accs:.3f}, lr: {scheduler._last_lr[-1]:.5f}')
        pbar.display()

    pbar.close()
    model.eval()


def make_weights_for_balanced_classes(cur_train_y: torch.Tensor):

    count = torch.unique(cur_train_y, return_counts=True)
    N = len(cur_train_y)
    weights = torch.zeros_like(cur_train_y, dtype=torch.float32)
    for cl, num in zip(*count):
        weights[cur_train_y == cl] = N / num

    return torch.Tensor(weights)

def evaluate(train_x, train_y, val_x, val_y, model=None, classifier=KNeighborsClassifier(metric='euclidean', weights='distance'),
             per_class_metric=False, return_cf=False, return_time=False):

    start = time.time()
    if model is not None:
        model.eval()

        with torch.no_grad():
            train_x = model(train_x.cuda()).cpu()
            val_x = model(val_x.cuda()).cpu()

    classifier.fit(train_x, train_y)
    preds = classifier.predict(val_x)
    dur = time.time() - start

    ret = []
    if per_class_metric:
        matrix = confusion_matrix(val_y, preds)
        per_class_accs = matrix.diagonal()/matrix.sum(axis=1)
        ret.append(torch.Tensor(per_class_accs))
    else:
        ret.append(torch.Tensor(accuracy_score(preds, val_y)))

    if return_cf:
        cm = confusion_matrix(val_y, preds.astype(int))

        ret = [ret, cm]

    if return_time:
        ret.append(dur)

    return ret if len(ret) > 1 else ret[0]