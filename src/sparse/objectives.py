import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from ..utils import uniform_decompose


class Objective:
    """
    Base class for an optimization functional.
    """
    def __init__(self, A, b, num_nodes, device='cpu'):
        """
        Params:
        -------
            A: Scipy CSR matrix
                Needs documentation.

            b: Numpy 1D array 
                Needs documentation.
        """
        chunk_sizes = uniform_decompose(A.shape[0], num_nodes)
        slices = np.cumsum(np.r_[0, chunk_sizes])
        block_size = max(chunk_sizes)

        data = A.data
        rows = A.tocoo().row
        cols = A.tocoo().col
        A_3d = []

        for i in range(num_nodes):
            ids = A.indptr[slices[i]: slices[i+1]+1]
            data_i = data[ids[0]: ids[-1]]
            rows_i = rows[ids[0]: ids[-1]]
            cols_i = cols[ids[0]: ids[-1]]
            rows_i -= rows_i[0]

            A_i = torch.sparse_coo_tensor(
                    (rows_i, cols_i), torch.Tensor(data_i),
                    size=(block_size, A.shape[1]), device=device)
            A_3d.append(A_i)

        b = torch.Tensor(b, device=device)
        self.b = pad_sequence(b.split(chunk_sizes), batch_first=True)
        self.A = torch.stack(A_3d).coalesce()


class LeastSquares(Objective):
    def __call__(self, X):
        if X.ndim == 2: X = X[..., None]
        else: X = X.expand(self.A.shape[0],-1)[...,None]

        Y = self.A.bmm(X).squeeze(-1) - self.b
        return Y.square().sum()


class LogRegression(Objective):
    def __call__(self, X):
        if X.ndim == 2: X = X[..., None]
        else: X = X.expand(self.A.shape[0],-1)[...,None]

        Y = self.A.bmm(X).squeeze(-1)
        Y = torch.logaddexp(-self.b*Y, Y.new([0.])).mean()
        return Y
