# -*- coding: utf-8 -*-
#
#  Copyright 2021 Timur Gimadiev <timur.gimadiev@gmail.com>
#  This file is part of neuralGTM.
#
#  neuralGTM is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from torch import pca_lowrank, tensor, float64, cdist, diag, exp, eye, log, pinverse, matmul
from torch import solve


class tGTM(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_rbfs=10, sigma=1, alpha=1e-3, n_grids=20, method='mean',
                 max_iter=10, tol=1e-5, random_state=None, verbose=False, use_pca_sklearn=False, center=False):
        self.n_components = n_components
        self.n_rbfs = n_rbfs
        self.sigma = tensor(sigma, dtype=float64)
        self.alpha = tensor(alpha, dtype=float64)
        self.n_grids = n_grids
        self.max_iter = max_iter
        self.method = method
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.prev_likelihood_ = -float('inf')
        self.pca = use_pca_sklearn
        self.center = center

    def get_lattice_points(self, n_grid):
        grid = np.meshgrid(*[np.linspace(-1, 1, n_grid + 1) for _ in range(self.n_components)])
        return np.array([c.ravel() for c in grid]).T

    def init(self, X):
        # generate map
        self.z = tensor(self.get_lattice_points(self.n_grids), requires_grad=False)
        self.rbfs = tensor(self.get_lattice_points(self.n_rbfs), requires_grad=False)
        d = cdist(self.z, self.rbfs, p=2).square_()
        self.phi = exp(-d / (2 * self.sigma))
        # init W and beta from PCA
        if not self.pca:  #use torch oca
            _, explained_variance,  components = pca_lowrank(X, center=self.center, q=X.size()[1])
        else:
            pca = PCA(n_components=self.n_components + 1, random_state=11)
            pca.fit(X)
            components = tensor(pca.components_)
            explained_variance = tensor(pca.explained_variance_)
        self.W = matmul(matmul(pinverse(self.phi), self.z), components[:self.n_components, :])
        betainv1 = explained_variance[self.n_components]
        inter_dist = cdist(matmul(self.phi, self.W), matmul(self.phi, self.W), p=2)
        inter_dist.fill_diagonal_(np.inf)
        betainv2 = inter_dist.min(axis=0).values.mean() / 2
        self.beta = 1 / max(betainv1, betainv2)

    def responsibility(self, X):
        p = exp((-self.beta / 2) * cdist(self.phi @ self.W, X, p=2).square_())
        return p / p.sum(axis=0)

    def likelihood(self, X):
        R = self.responsibility(X)
        D = X.size()[1]
        k1 = (D / 2) * log(self.beta / (2 * np.pi))
        k2 = -(self.beta / 2) * cdist(matmul(self.phi, self.W), X, p=2).square_()
        return (R * (k1 + k2)).sum()

    def fit(self, X, y=None, **fit_params):
        return self.partial_fit(X)

    def partial_fit(self, X, y=None, **fit_params):
        if not hasattr(self, 'W'):
            self.init(X)
        for i in range(self.max_iter):
            R = self.responsibility(X)
            G = diag(R.sum(axis=1))
            B = matmul(matmul(self.phi.T, R), X)
            A = matmul(matmul(self.phi.T, G), self.phi) + (self.alpha / self.beta) * eye(self.phi.size()[1])
            self.W, _ = solve(B, A)
            self.beta = X.nelement() / (cdist(matmul(self.phi, self.W), X, p=2).square_() * R).sum()
            likelihood = self.likelihood(X)
            diff = abs(likelihood - self.prev_likelihood_) / X.size()[0]
            self.prev_likelihood_ = likelihood
            if self.verbose:
                print('cycle #{}: likelihood: {:.3f}, diff: {:.3f}'.format(i + 1, likelihood, diff))
            if diff < self.tol:
                if self.verbose:
                    print('converged.')
                    break
        return self

    def transform(self, X, y=None):
        assert self.method in ('mean', 'mode')
        if self.method == 'mean':
            R = self.responsibility(X)
            return (matmul(self.z.T, R)).T
        elif self.method == 'mode':
            return self.z[self.responsibility(X).argmax(axis=0), :]

    def inverse_transform(self, Xt):
        d = cdist(Xt, self.rbfs, p=2).square_()
        phi = exp(-d / (2 * self.sigma))
        return matmul(phi, self.W)


__all__ = ['tGTM']
