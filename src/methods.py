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
                as one of the output variables
                must produce a mixing matrix of a generated graph
        """
        self.F = F
        self.alpha = alpha
        
        self.n = F.b.size(0)
        self.gen = graph_generator
        self._initLogs()
        self._k = 0

    def _args(self, kwargs):
        raise NotImplementedError
        
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

    def run(self, X0, n_iters=10, lp=1, **kwargs):
        if kwargs: args = self._args(kwargs)
        else: args = self._stateInit(X0)

        for k in range(n_iters):
            X0, *args = self._step(X0, *args)
            if k%lp == 0: self._record(X0, k+self._k)
        self._k += k

        return X0, *args
        
        
class EXTRON(DGM):
    """
    ONe-process EXTRA algorithm
    """
    def _args(self, kwargs):
        G0 = kwargs['G0']
        X1 = kwargs['X1']
        return G0, X1

    def _stateInit(self, X0):
        _, W = self._G, self._W = self.gen()
        X0.requires_grad_(True)
        G0 = D(self.F(X0), X0)
        with torch.no_grad():
            X1 = X0@W - self.alpha*G0
        self._initLogs()
        self._record(X1, 0)
        self._k += 1

        return G0, X1

    def _step(self, X0, G0, X1):
        _,W = self._G,self._W = self.gen()
        X1.requires_grad_(True)
        G1 = D(self.F(X1), X1)
        with torch.no_grad():
            X2 = X1 - X0/2 + (X1-X0/2)@W - self.alpha*(G1-G0)

        return X1, G1, X2


class DIGONing(DGM):
    """
    ONe-process DIGing algorithm
    """
    def _args(self, kwargs):
        G0 = kwargs['G0']
        Y0 = kwargs['Y0']
        return G0, Y0

    def _stateInit(self, X0):
        X0.requires_grad_(True)
        Y0 = D(self.F(X0), X0)
        G0 = Y0.clone()
        return G0, Y0

    def _step(self, X0, G0, Y0):
        _, W = self._G, self._W = self.gen()
        with torch.no_grad():
            X1 = X0@W - self.alpha*Y0
            
        X1.requires_grad_(True)
        G1 = D(self.F(X1), X1)
        with torch.no_grad():
            Y1 = Y0@W + G1 - G0

        return X1, G1, Y1
    
    
class DAGDON(DGM):
    """
    Decentralized ONe-process AGD subroutine
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
        super().__init__(F, graph_generator)
        del self.alpha

        self.L = L
        self.mu = mu
        self.consensus = TensorAccumulator(T)

    def _args(self, kwargs):
        A0 = kwargs['A0']
        U0 = kwargs['U0']
        return A0, U0

    def _stateInit(self, X0):
        self._initLogs()
        A0, U0 = 0, X0.clone()
        return A0, U0

    def _step(self, X0, A0, U0):
        _, W = self._G, self._W = self.gen()
        self.consensus.append(W)
        
        a = 1 + A0*self.mu
        a = (a + (a**2 + 4*self.L*A0*a)**.5)/(2*self.L)
        A1 = A0 + a

        Y = (a*U0 + A0*X0) / A1
        Y.requires_grad_(True)
        G = D(self.F(Y), Y)

        with torch.no_grad():
            V = self.mu*Y + (1+A0*self.mu)*U0 - a*G
            V /= 1 + A0*self.mu + self.mu
            U1 = self.consensus.mm(V)
            X1 = (a*U1 + A0*X0) / A1

        return X1, A1, U1
