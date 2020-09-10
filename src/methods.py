import torch
from .utils import D


class DGMBase:
    """
    Base class for decentralized gradient methods.
    """
    def __init__(self, F, graph_generator, alpha=1.):
        """
        Params:
        -------
            F: Objective
                Objective to optimize.

            alpha: float
                Step size in the gradients' update.

            graph_generator: object
                As one of the output variables
                must produce a mixing matrix of a generated graph.
        """
        self.F = F
        self.alpha = self._alpha = alpha
        
        self.n = F.b.size(0)
        self.gen = graph_generator
        self._initLogs()

    def _args(self, kwargs):
        raise NotImplementedError

    def _stateInit(self, X0):
        raise NotImplementedError

    def _step(self, X0, *args):
        raise NotImplementedError

    def _updateStepsize(self, k):
        pass
        
    def _initLogs(self):
        self.logs = {'i': [], 'fn': [], 'dist2con': []}
        self._k = 0
        
    def _dist2consensus(self, X):
        return X.var(dim=0, unbiased=False).sum() * self.n
        
    def _record(self, X, k):
        self.logs['dist2con'].append(self._dist2consensus(X).item())
        self.logs['fn'].append(self.F(X.mean(0)).item())
        self.logs['i'].append(k)

    def run(self, X0, n_iters=100, lp=1, **kwargs):
        if kwargs: args = self._args(kwargs)
        else: args = self._stateInit(X0)

        for k in range(self._k, self._k+n_iters):
            X0, *args = self._step(X0, *args)
            if k%lp == 0: self._record(X0, k)
            self._updateStepsize(k)
        self._k += n_iters

        ## Next line works only in python >= 3.8
        # return X0, *args

        ## For compatibility with python <= 3.7,
        ## variable `out` is imposed
        out = X0, *args
        return out
        
        
class EXTRA(DGMBase):
    """
    One-process EXTRA algorithm
    """
    def _args(self, kwargs):
        G0 = kwargs['G0']
        X1 = kwargs['X1']
        return G0, X1

    def _stateInit(self, X0):
        W = self.gen()
        X0.requires_grad_(True)
        G0 = D(self.F(X0), X0)
        with torch.no_grad():
            X1 = W@X0 - self.alpha*G0
        self._initLogs()
        self._record(X1, 0)
        self._k += 1

        return G0, X1

    def _step(self, X0, G0, X1):
        W = self.gen()
        X1.requires_grad_(True)
        G1 = D(self.F(X1), X1)
        with torch.no_grad():
            X2 = X1 - X0/2 + W@(X1-X0/2) - self.alpha*(G1-G0)

        return X1, G1, X2


class DIGing(DGMBase):
    """
    One-process DIGing algorithm
    """
    def _args(self, kwargs):
        G0 = kwargs['G0']
        Y0 = kwargs['Y0']
        return G0, Y0

    def _stateInit(self, X0):
        self._initLogs()
        X0.requires_grad_(True)
        Y0 = D(self.F(X0), X0)
        G0 = Y0.clone()
        return G0, Y0

    def _step(self, X0, G0, Y0):
        W = self.gen()
        with torch.no_grad():
            X1 = W@X0 - self.alpha*Y0
            
        X1.requires_grad_(True)
        G1 = D(self.F(X1), X1)
        with torch.no_grad():
            Y1 = W@Y0 + G1 - G0

        return X1, G1, Y1
    
    
class DAccGD(DGMBase):
    """
    Decentralized one-process AGD subroutine
    """
    def __init__(self, F, graph_generator, L=1., mu=0., T=20):
        """
        Params:
        -------
            L: float
                Average of Lipshitz constants in the objective.

            mu: float
                Average of strong convexity constants. If mu > 0,
                float overflow occurs after some number of iterations.

            T: int
                Number of consequently generated graphs
                to use in the consensus operation.
        """
        super().__init__(F, graph_generator)
        del self.alpha

        self.L = L
        self.mu = mu
        self.T = T

    def _args(self, kwargs):
        A0 = kwargs['A0']
        U0 = kwargs['U0']
        return A0, U0

    def _stateInit(self, X0):
        self._initLogs()
        A0, U0 = 0, X0.clone()
        return A0, U0

    def _step(self, X0, A0, U0):
        a = 1 + A0*self.mu
        a = (a + (a**2 + 4*self.L*A0*a)**.5) / (2*self.L)
        A1 = A0 + a

        Y = (a*U0 + A0*X0) / A1
        Y.requires_grad_(True)
        G = D(self.F(Y), Y)

        W_series = [self.gen() for _ in range(self.T)]
        with torch.no_grad():
            V = self.mu*Y + (1+A0*self.mu)*U0 - a*G
            V /= 1 + A0*self.mu + self.mu
            U1 = torch.chain_matmul(*W_series, V)
            X1 = (a*U1 + A0*X0) / A1

        return X1, A1, U1


class DSGD(DGMBase):
    """
    Decentralized one-process (Gossip) SGD
    """
    def _args(self, kwargs):
        return ()

    def _stateInit(self, X0):
        self._initLogs()
        return ()

    def _updateStepsize(self, k):
        self.alpha = self._alpha # / (1+k)**.5

    def _step(self, X0, *args):
        X0.requires_grad_(True)
        G = D(self.F(X0), X0)
        W = self.gen()
        with torch.no_grad():
            X1 = W @ (X0-self.alpha*G)
        return (X1,)
