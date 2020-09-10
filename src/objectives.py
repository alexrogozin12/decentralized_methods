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
        i = '' if X.ndim < 2 else 'i'
        Y = torch.einsum(f'ijk,{i}k->ij', self.A, X) - self.b
        return Y.square().mean()


class LogRegression(Objective):
    def __call__(self, X):
        i = '' if X.ndim < 2 else 'i'
        Y = torch.einsum(f'ij,ijk,{i}k->ij', self.b, self.A, X)
        Y = torch.logaddexp(-Y, Y.new([0.])).mean()
        return Y


class StochObjective(Objective):
    """
    Base class for a stochastic optimization functional.

    Params:
    -------
        m: torch.tensor
            Number of rows selected in a block.

        avg: float
            Number of batches sampled on every node.

        mCols: torch.LongTensor
            Initial batch sizes on nodes.

        static: bool
            If set to True, freezes the batch sizes to ones
            given at the initialization.
    """
    def __init__(self, A, b, num_nodes, avg=1, mCols=None, static=True):
        super().__init__(A, b, num_nodes)
        self.m = mCols
        self.avg = avg

        n, d = self.b.shape
        self.static = static
        self.sampleM = lambda : self._sampleM(d//2, d, n)
        if mCols is None and static: self.m = self.sampleM()

    def _sampleXi(self, m):
        ones = torch.ones_like(self.b)
        fill = torch.multinomial(ones, ones.size(1))

        xi = self.b.new_tensor(fill)
        xi = (xi // m[:, None] < 1.).float()
        return xi

    def sampleXi(self, m):
        xi = self.b.new(self.avg, *self.b.shape)
        for r in xi: r[:] = self._sampleXi(m)
        return xi

    def _sampleM(self, limA, limB, size):
        return torch.randint(limA, limB, (size,))

    def __call__(self, X):
        if self.static: self.m = self.sampleM()
        return self._fnCall(X, self.m)

    def _fnCall(self, X, m):
        raise NotImplementedError


class StochLeastSquares(StochObjective):
    def _fnCall(self, X, m):
        xi = self.sampleXi(m)
        i = '' if X.ndim < 2 else 'i'
        p = '' if xi.ndim < 3 else 'p'
        Y = torch.einsum(
                f'{p}ij, ijk,{i}k->{p}ij',
                xi, self.A, X) - xi*self.b

        if p: Y = Y.mean(0)
        return Y.square().mean()


class StochLogRegression(StochObjective):
    def _fnCall(self, X, m):
        xi = self.sampleXi(m)
        xi *= self.b

        i = '' if X.ndim < 2 else 'i'
        p = '' if xi.ndim < 3 else 'p'
        Y = torch.einsum(f'{p}ij,ijk,{i}k->{p}ij', xi, self.A, X)
        if p: Y = Y.mean(0)

        Y = torch.logaddexp(-Y, Y.new([0.])).mean()
        return Y
