import math
import torch
from .utils import D, lambda_2, expected_lambda2


class DGMBase:
    """
    Base class for decentralized gradient methods.
    """
    def __init__(self, F, graph_generator, eta=1.):
        """
        Params:
        -------
            F: Objective
                Objective to optimize.

            eta: float
                Step size in the gradients' update.

            graph_generator: object
                As one of the output variables
                must produce a mixing matrix of a generated graph.
        """
        self.F = F
        self.eta = self._eta = eta
        self.gen = graph_generator
        self.con_iters = 1

    def _args(self, kwargs):
        raise NotImplementedError

    def _stateInit(self, X0):
        raise NotImplementedError

    def _step(self, X0, *args):
        raise NotImplementedError

    def _consensus(self, X0, *args):
        cT = self._cT + self.con_iters
        self._cT = cT

        self.logs['nmix'].append(cT)
        return self._consensusUpdate(X0, *args)

    def _consensusUpdate(self, X0):
        W = self.gen()
        X1 = W @ X0
        return X1

    def _updateStepsize(self, k):
        pass
        
    def _initLogs(self):
        self.logs = {'i': [], 'fn': [], 'dist2con': [], 'nmix': []}
        self._cT = 0
        self._k = 0
        
    def _dist2consensus(self, X):
        return X.var(dim=0, unbiased=False).sum() * self.F.b.size(0)
        
    def _record(self, X, k):
        self.logs['dist2con'].append(self._dist2consensus(X).item())
        self.logs['fn'].append(self.F(X.mean(0)).item())
        self.logs['i'].append(k)

    # FIXME: add ability to continue from args
    def run(self, X0, n_iters=100, lp=1, **kwargs):
        """
        Params:
        -------
            X0: torch.Tensor of shape (n, d),
                n - num of nodes (agents), d - data dimensionality.
        """
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
        self._initLogs()
        X0.requires_grad_(True)
        G0 = D(self.F(X0), X0)
        with torch.no_grad():
            Y = self._consensus(X0)
            X1 = Y - self.eta*G0
        self._record(X1, 0)
        self._k += 1

        return G0, X1

    def _step(self, X0, G0, X1):
        X1.requires_grad_(True)
        G1 = D(self.F(X1), X1)
        with torch.no_grad():
            Y0 = X1 - .5*X0
            Y1 = self._consensus(Y0)
            X2 = Y0 + Y1 - self.eta*(G1-G0)

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
        with torch.no_grad():
            WX, WY = self._consensus(X0, Y0)
            X1 = WX - self.eta*Y0
            
        X1.requires_grad_(True)
        G1 = D(self.F(X1), X1)
        with torch.no_grad():
            Y1 = WY + G1 - G0

        return X1, G1, Y1

    def _consensusUpdate(self, X0, Y0):
        W = self.gen()
        X1 = W @ X0
        Y1 = W @ Y0
        return X1, Y1


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
        self.eta = self._eta # / (1+k)**.5

    def _step(self, X0, *args):
        X0.requires_grad_(True)
        G = D(self.F(X0), X0)
        W = self.gen()
        with torch.no_grad():
            X1 = W @ (X0-self.eta*G)
        return (X1,)

    
class DAccGD(DGMBase):
    """
    Decentralized one-process AGD subroutine
    """
    def __init__(self, F, graph_generator, L, mu, con_iters=20):
        """
        Params:
        -------
            L: float
                Average of Lipshitz constants in the objective.

            mu: float
                Average of strong convexity constants. If mu > 0,
                float overflow occurs after some number of iterations.

            con_iters: int
                Number of consequently generated graphs
                to use in the consensus operation.
        """
        super().__init__(F, graph_generator)
        del self.eta

        self.L = L
        self.mu = mu
        self.con_iters = con_iters

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

        with torch.no_grad():
            V = self.mu*Y + (1+A0*self.mu)*U0 - a*G
            V /= 1 + A0*self.mu + self.mu
            U1 = self._consensus(V)
            X1 = (a*U1 + A0*X0) / A1

        return X1, A1, U1

    def _consensusUpdate(self, V):
        W_series = [self.gen() for _ in range(self.con_iters)]
        U1 = torch.chain_matmul(*W_series, V)
        return U1


class Mudag(DGMBase):
    """
    Implementation of arXiv:2005.00797 (Mudag)

    Params:
    -------
        M: float
            Characterizes M-smoothness of the problem

        kappa_g: float
            Condition number of the objective
    """
    def __init__(self, F, W, L, mu, M, kappa_g, scale=.01):
        super().__init__(F, None, 1./L)
        gamma = math.log(M/L*kappa_g)
        self._alpha = self._setAlpha(mu, L)
        self._setConsensus(W, gamma, scale)

    def _setConsensus(self, W, gamma, scale):
        self.W = W
        s2 = lambda_2(W)
        assert s2 < 1

        assert gamma > 1
        gamma *= scale

        eta_w = (1 - s2*s2)**.5
        self.eta_w = (1-eta_w) / (1+eta_w)
        self.K = math.ceil(1./(1-s2)**.5 * gamma)

    def _setAlpha(self, mu, L):
        alpha = (mu/L)**.5
        alpha = (1-alpha) / (1+alpha)
        return alpha

    def _args(self, kwargs):
        Y0 = kwargs['Y0']
        G0 = kwargs['G0']
        Y1 = kwargs['Y1']
        return Y0, G0, Y1

    def _stateInit(self, X0):
        self._initLogs()
        X0 = X0.clone().mean(0).expand(len(X0), -1)
        Y0 = X0.clone()
        G0 = torch.zeros_like(Y0)
        Y1 = X0.clone()
        return Y0, G0, Y1

    def _step(self, X0, Y0, G0, Y1):
        Y1.requires_grad_(True)
        G1 = D(self.F(Y1), Y1)

        with torch.no_grad():
            X1 = self._consensus(Y1+X0-Y0-self.eta*(G1-G0))
            Y2 = X1 + self._alpha*(X1-X0)

        return X1, Y1, G1, Y2

    def _consensusUpdate(self, X0):
        """
        FastMix update - an efficient way of averaging
        proposed by Liu & Morse (2011).
        """
        X1 = X0.clone()
        for _ in range(self.K):
            X2 = (1+self.eta_w)*self.W@X1 - self.eta_w*X0
            X0, X1 = X1, X2

        return X1


class SDAccGD(DAccGD):
    """
    DAccGD with FastMix-like update
    """
    def __init__(self, F, graph_generator, L, mu, E_s2, con_iters):
        super().__init__(F, graph_generator, L, mu, con_iters)
        self.eta_w = self._setEtaW(E_s2)

    def _setEtaW(self, E_s2):
        eta_w = (1 - E_s2*E_s2)**.5
        eta_w = (1-eta_w) / (1+eta_w)
        return eta_w

    def _consensusUpdate(self, X0):
        """
        An adaptation of FastMix for time-varying graphs.
        """
        X1 = X0.clone()
        for _ in range(self.con_iters):
            W = self.gen()
            X2 = (1+self.eta_w)*W@X1 - self.eta_w*X0
            X0, X1 = X1, X2

        return X1


class SMudag(Mudag, SDAccGD):
    """
    Stochastic adaptation (with mixing matrix
    update per consensus iteration) of Mudag algorithm.
    """
    def __init__(self, F, graph_generator, L, mu, E_s2, con_iters):
        DGMBase.__init__(self, F, graph_generator, 1./L)
        self.eta_w = SDAccGD._setEtaW(self, E_s2)
        self._alpha = self._setAlpha(mu, L)
        self.con_iters = con_iters

    def _consensusUpdate(self, X0):
        X1 = SDAccGD._consensusUpdate(self, X0)
        return X1


class APM_C(DGMBase):
    """
    Original implementation of Accelerated Penalty Method with Consensus.
    (arXiv.org > math > arXiv:1810.01053)
    """
    def __init__(self, F, W, L, mu, beta, scale=.01):
        super().__init__(F, None, 1./L)
        self._setEssential(L, mu, beta)
        self._setConsensus(W, scale)

    def _setConsensus(self, W, scale):
        self.W = W
        s2 = lambda_2(W)
        assert s2 < 1

        eta_w = (1 - s2*s2)**.5
        self.eta_w = (1-eta_w) / (1+eta_w)
        self._gamma = scale*self.theta / (1-s2)**.5

    def _setEssential(self, L, mu, beta):
        theta = (mu/L)**.5
        alpha = (1-theta) / theta
        alpha *= (L*theta-mu) / (L-mu)

        self.theta = theta
        self._alpha = alpha
        self.beta = beta

    def _stateInit(self, X0):
        self._initLogs()
        X1, k = X0.clone(), 0
        return X1, k

    def _args(self, kwargs):
        X1 = kwargs['X1']
        k = kwargs['k']
        return X1, k

    def _step(self, X0, X1, k):
        Y = X1 + self._alpha*(X1-X0)
        Y.requires_grad_(True)
        G = D(self.F(Y), Y)

        with torch.no_grad(): Z0 = Y - self.eta*G
        Z1 = self._consensus(Z0, k)

        upsilon = (1 - self.theta)**(k+1)
        X2 = upsilon*Z1 + self.beta*Z0
        X2 = X2 / (upsilon + self.beta)
        return X1, X2, k+1

    def _consensusUpdate(self, Z0, k):
        T_k = math.ceil(k*self._gamma)

        Z1 = Z0.clone()
        for _ in range(T_k):
            Z2 = (1+self.eta_w)*self.W@Z1 - self.eta_w*Z0
            Z0, Z1 = Z1, Z2

        return Z1


class SAPM_C(APM_C):
    """
    An adaptation of APM-C for time-varying graphs.
    """
    def __init__(self, F, graph_generator, L, mu, beta, E_s2, scale=.01):
        DGMBase.__init__(self, F, graph_generator, 1./L)
        self._setEssential(L, mu, beta)
        self._setConsensus(E_s2, scale)

    def _setConsensus(self, E_s2, scale):
        eta_w = (1 - E_s2*E_s2)**.5
        self.eta_w = (1-eta_w) / (1+eta_w)
        self._gamma = self.theta/(1-E_s2)**.5
        self._gamma *= scale

    def _consensusUpdate(self, Z0, k):
        self.con_iters = math.ceil(k*self._gamma)

        Z1 = Z0.clone()
        for _ in range(self.con_iters):
            W = self.gen()
            Z2 = (1+self.eta_w)*W@Z1 - self.eta_w*Z0
            Z0, Z1 = Z1, Z2

        return Z1
