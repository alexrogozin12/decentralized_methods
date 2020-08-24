import torch

from torch.nn.utils.rnn import pad_sequence
from .utils import uniform_decompose


class Objective:
    """
    Base class for an optimization functional
    """
    def __init__(self, A, b, num_nodes):
        """
        A - matrix, i.e. torch 2D tensor
        b - vector, i.e. torch 1D tensor
        """
        chunk_sizes = uniform_decompose(A.size(0), num_nodes)
        self.A = pad_sequence(A.split(chunk_sizes), batch_first=True)
        self.b = pad_sequence(b.split(chunk_sizes), batch_first=True)


class LeastSquares(Objective):
    def __call__(self, X):
        s = '' if X.ndim < 2 else 'i'
        Y = torch.einsum(f'ijk,k{s}->ij', self.A, X) - self.b
        return Y.square().sum()


class LogRegression(Objective):
    def __call__(self, X):
        s = '' if X.ndim < 2 else 'i'
        Y = torch.einsum(f'ij,ijk,k{s}->ij', self.b, self.A, X)
        Y = torch.logaddexp(-Y, Y.new([0.])).mean()
        return Y
