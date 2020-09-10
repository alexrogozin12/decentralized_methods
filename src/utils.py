import math
import numpy as np

import torch
import networkx as nx

from torch import autograd
from torch_geometric.utils.random import erdos_renyi_graph


def uniform_decompose(N, m, b=8):
    """
    Decomposes N into m terms: a_i, i=1,m;
    so that: max_i a_i - min_i a_i <= 1 applies
    --------
    b is a number of bits used for an integer
    in the output array.
    """
    terms = np.empty(m, dtype=f'i{b}')
    terms[:] = math.floor(N/m)
    terms[:N-terms.sum()] += 1
    return terms.tolist() 


def D(y, x):
    """
    Differential operator.
    """
    grad = autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, allow_unused=True)

    if len(grad) == 1:
        return grad[0]
    return grad


def metropolis_weights(A):
    """
    A - Adjacency matrix (i.e., symmetric and with zero diagonal).
    """
    A = A / (1 + torch.max(A.sum(1,keepdims=True),A.sum(0,keepdims=True)))
    A.as_strided([len(A)], [len(A)+1]).copy_(1-A.sum(1))
    return A


class PythonGraph:
    """
    Graph-generator.
    Returns a generated graph and its mixing matrix.

    Params:
        F: Objective
            Functional being optimized that keeps information
            about the number of nodes and data type to work with.

        graph: str
            Type of a graph to generate.

        p: float, non-negative
            In random graphs, probability that an edge exists.
            The value is ignored in case of deterministic graphs.
    """
    def __init__(self, F, graph='erdos_renyi', pr=.2):
        num_nodes = F.b.size(0)

        if graph == 'erdos_renyi':
            self.args = (num_nodes, pr)
        elif graph == 'random_geometric':
            graph = 'generators.geometric.' + graph
            self.args = (num_nodes, pr)
        else:
            self.args = (num_nodes,)
        self.method = eval(f'nx.{graph}_graph')
        self._mold = F.b.new_tensor

    def gen(self):
        G = self.method(*self.args)
        S = nx.adjacency_matrix(G)
        S = self._mold(S.todense())
        W = metropolis_weights(S)
        return G, W


def consensus_variation(F, p=.2):
    """
    Graph-generator.
    Returns `erdos-renyi` graph and its mixing matrix.
    This function requires `torch_geometric` installed

    Params:
        F: Objective
            Functional being optimized that keeps information
            about the number of nodes and data type to work with.

        p: float, non-negative
            In random graphs, probability that an edge exists.
            The value is ignored in case of deterministic graphs.
    """
    num_nodes = F.b.size(0)
    e1, e2 = erdos_renyi_graph(num_nodes, p)
    S = F.b.new_zeros(num_nodes, num_nodes)
    S[e1, e2] = 1.

    W = metropolis_weights(S)
    return W


class ConsensusAmplifier:
    """
    Generates and accumulates/stores mixing matrices

    Params:
    -------
        gen: function
            Generates a graph and returns its mixing matrix.

        seq_length: int, positive
            Number of mixing matrices in accumulator list.

        stingy: bool
            Update mode. If True, the oldest mixing matrix
            is dropped and a new one is added.
    """
    def __init__(self, gen, seq_length, stingy=True, W_series=[]):
        self.gen = gen
        self.n = seq_length
        if stingy:
            self.acc = W_series
            assert(len(W_series) <= seq_length)
            self.update = self._update_old
        else:
            self.update = self._update_all

    def _update_all(self):
        self.acc = []
        for _ in range(self.n):
            W = self.gen()
            self.acc.append(W)
        return W

    def _update_old(self):
        W = self.gen()
        if len(self.acc) >= self.n:
            self.acc.pop(0)
        self.acc.append(W)
        return W
            
    def mm(self, X):
        """
        Performs efficient multiplication of
        mixing matrices and given tensor.
        """
        return torch.chain_matmul(*self.acc, X)
