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

import pandas as pd

import numpy as np
import torch
import os
from scipy import stats as s
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import SGDOneClassSVM
from sklearn.svm import OneClassSVM


class classifier:
    def __init__(self, knn_file=None, savefile=None, save_to_file=True, **kwargs):

        self.x_data = None
        self.y_data = None
        self.save_file = knn_file if not savefile else savefile
        self.classes = []

        self.is_fitted = False

        self.save_to_file = save_to_file

        self.model = KNeighborsClassifier(
            n_neighbors=10, weights='distance', metric='euclidean')

        # self.model = LogisticRegression()
        self.le = LabelEncoder()

        # self.outlier_detector = OneClassSVM(nu=0.1)
        self.outlier_detector = LocalOutlierFactor(
            novelty=True, metric='cosine', n_neighbors=5)

        if knn_file:
            print(f'loading data from file: {knn_file}')
            if (os.path.exists(knn_file)):
                print('File found')
                data = torch.load(knn_file)
                self.add_points(data['x'], data['y'])

                print(
                    f'Found {self.x_data.shape[0]} points with {len(set(self.y_data))} classes')
                print(pd.Series(self.y_data).value_counts())
                self.is_fitted = True

            else:
                print('File not found')

    def print_info(self):
        print(pd.Series(self.y_data).value_counts())

    def add_points(self, x, y):
        if self.x_data is None:
            self.x_data = np.array(x)
            self.y_data = y
        else:
            self.x_data = np.concatenate([self.x_data, x])
            self.y_data = self.y_data + y

        self.classes = list(set(self.y_data))
        self.label_data = self.le.fit_transform(self.y_data)
        self.outlier_detector.fit(self.x_data)
        # print(self.outlier_detector.offset_)
        self.model.fit(self.x_data, self.label_data)

        if self.save_to_file:
            torch.save({'x': self.x_data,
                        'y': self.y_data}, self.save_file)
        self.is_fitted = True

    def remove_class(self, cl):
        inds_to_keep = [idx for idx, el in enumerate(self.y_data) if el != cl]

        self.x_data = self.x_data[inds_to_keep]
        self.y_data = [self.y_data[i] for i in inds_to_keep]

        self.classes = list(set(self.y_data))

        self.outlier_detector.fit(self.x_data)
        self.model.fit(self.x_data, self.y_data)
        if self.save_to_file:
            torch.save({'x': self.x_data,
                        'y': self.y_data}, self.save_file)

    def classify(self, x):
        if not self.is_fitted:
            return None, None, None
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        probs = self.model.predict_proba(x)
        max_ids = np.argmax(probs, axis=1)

        classes = self.le.inverse_transform(max_ids)
        confs = np.max(probs, axis=1)



        outliers = self.outlier_detector.predict(x)
        scores = self.outlier_detector.decision_function(x)

        # print(outliers)

        # D = np.array([1000.0 if o == -1 else 0.0 for o in outliers])

        D = self.model.kneighbors(x, 1)[0][:, 0]
    
        # D = scores
        # D = [np.nan] * len(classes)

        return classes, confs, D
