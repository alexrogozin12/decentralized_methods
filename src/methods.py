import torch
from .utils import D, dummy_consensus_variation, TensorAccumulator


class DGM:
    """
    Base class for decentralized gradient methods
    """
    def __init__(self, F, graph_generator, alpha=1.):
        """
        Params:
        -------
            F: Objective
                objective to optimize

            alpha: float
                step size in the gradients' update

            graph_generator: object
                must produce a mixing matrix of a generated graph
        """
        self.F = F
        self.alpha = alpha
        
        self.n = F.b.size(0)
        self.gen = graph_generator
        self._initLogs()
        
    def _initLogs(self):
        self.logs = {'i': [], 'fn': [], 'dist2con': []}
        
    def _dist2consensus(self, X):
        h = X.new(self.n).fill_(1.)
        Q = torch.norm(X - X/self.n @ h[:,None]*h)
        return Q
        
    def _record(self, X, k):
        self.logs['dist2con'].append(self._dist2consensus(X).item())
        self.logs['fn'].append(self.F(X.mean(1)).item())
        self.logs['i'].append(k)
        
        
class EXTRON(DGM):
    """
    ONe-process EXTRA algorithm
    """
    def _step1(self, X0):
        _, W = self._G, self._W = self.gen()
        X0.requires_grad_(True)
        G0 = D(self.F(X0), X0)
        with torch.no_grad():
            X1 = X0@W - self.alpha*G0
        return G0, X1

    def _step2(self, X0, G0, X1):
        _, W = self._G, self._W = self.gen()
        X1.requires_grad_(True)
        G1 = D(self.F(X1), X1)
        with torch.no_grad():
            X2 = X1 - X0/2 + (X1-X0/2)@W - self.alpha*(G1-G0)
        return X1, G1, X2

    def run(self, X0, G0=None, X1=None, n_iters=10, lp=1):
        if G0 is None or X1 is None:
            G0, X1 = self._step1(X0)
            self._initLogs()
            self._record(X1, 0)

        for k in range(1, n_iters):
            X0, G0, X1 = self._step2(X0, G0, X1)
            if k%lp == 0: self._record(X1, k)

        return X0, G0, X1


class DIGONing(DGM):
    """
    ONe-process DIGing algorithm
    """
    def run(self, X0, G0=None, Y0=None, n_iters=10, lp=1):
        if G0 is None or Y0 is None:
            self._initLogs()
            X0.requires_grad_(True)
            Y0 = D(self.F(X0), X0)
            G0 = Y0.clone()
            
        for k in range(1, n_iters):
            _, W = self._G, self._W = self.gen()
            with torch.no_grad():
                X1 = X0@W - self.alpha*Y0
                
            X1.requires_grad_(True)
            G1 = D(self.F(X1), X1)
            with torch.no_grad():
                Y1 = Y0@W + G1 - G0
                
            X0, Y0, G0 = X1, Y1, G1
            if k%lp == 0: self._record(X0, k)
            
        return X0, G0, Y0
    
    
class DAGDON(DGM):
    """
    One-process AGD subroutine 
    """
    def __init__(self, F, graph_generator, L=1., mu=0., T=20):
        """
        Params:
        -------
            L: float
                average of Lipshitz constants in the objective

            mu: float
                average of strong convexity constants

            T: int
                number of consequently generated grpahs
                to use in the consensus operation
        """
        self.F = F
        self.T = T
        self.L = L
        self.mu = mu
        self.n = F.b.size(0)
        self.gen = graph_generator
        
    def run(self, X0, A0=None, n_iters=10, lp=1):
        if A0 is None:
            a0 = A0 = 0
            U0 = X0.clone()
            self._initLogs()
            
        consensus = TensorAccumulator(self.T)
        for k in range(1, n_iters):
            _, W = self._G, self._W = self.gen()
            consensus.append(W)
            
            a1 = 1 + A0*self.mu
            a1 = (a1 + (a1**2 + 4*self.L*A0*a1)**.5)/(2*self.L)
            A1 = A0 + a1
            
            Y = (a1*U0 + A0*X0)/A1
            Y.requires_grad_(True)
            G = D(self.F(Y), Y)
            with torch.no_grad():
                V = self.mu*Y + (1+A0*self.mu)*U0 - a1*G
                V /= 1 + A0*self.mu + self.mu
                U1 = consensus.mm(V)
                X1 = (a1*U1 + A0*X0) / A1
                
            X0, A0 = X1, A1
            if k%lp == 0: self._record(X0, k)
                
        return X0, A0
