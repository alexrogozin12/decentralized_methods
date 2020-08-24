import math
import numpy as np

import torch
import networkx as nx

from torch import autograd


def uniform_decompose(N, m, b=8):
    """
    Decomposes N into m terms: a_i, i=1,m;
    so that: max_i a_i - min_i a_i <= 1 applies
    --------
    b is a number of bits used for an integer
    in the output array
    """
    terms = np.empty(m, dtype=f'i{b}')
    terms[:] = math.floor(N/m)
    terms[:N-terms.sum()] += 1
    return terms.tolist() 


def D(y, x):
    """
    Differential operator
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
    A - Adjacency matrix (i.e., symmetric and with zero diagonal)
    """
    A = A / (1 + torch.max(A.sum(1,keepdims=True),A.sum(0,keepdims=True)))
    A.as_strided([len(A)], [len(A)+1]).copy_(1-A.sum(1))
    return A


def dummy_consensus_variation(F, graph='erdos_renyi', p=.2):
    """
    Graph-generator
    Returns a generated graph and its mixing matrix

    Params:
        F: Objective
            functional being optimized that keeps information
            about the number of nodes and data type to work with

        graph: str
            type of a graph to generate

        p: float, non-negative
            in random graphs, probability that an edge exists
            The value is ignored in case of deterministic graphs
    """
    num_nodes = F.b.size(0)
    args = (num_nodes, p) if graph == 'erdos_renyi' else (num_nodes,)
    G = eval(f'nx.{graph}_graph')(*args)
    
    S = nx.adjacency_matrix(G).todense()
    S = F.b.new_tensor(S)
    
    W = metropolis_weights(S)
    return G, W 


class TensorAccumulator:
    """
    Accumulates tensors and performs efficient
    chained matrix multiplication
    """
    def __init__(self, seq_length):
        self.n = seq_length
        self.acc = []
    
    def append(self, X):
        if len(self.acc) >= self.n:
            self.acc.pop(0)
        self.acc.append(X)
            
    def mm(self, X):
        return torch.chain_matmul(X, *self.acc)
