import pdb
from typing import Optional, Tuple
import torch
import torch.nn as nn

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Solves entropic regularization optimal transport using the Sinkhorn-Knopp algorithm.

    Implementation based on https://pot.readthedocs.io/en/stable/_modules/ot/bregman.html#sinkhorn_stabilized
    and on https://arxiv.org/abs/1803.00567 (Section 4.2).

    Supports batching by adding an extra first dimension.

    Note: A marginal (a or b) of None defaults to uniform.
    """
    def __init__(self, eps, max_iter, reduction='none', 
                unbalanced_lambda: Optional[float] = None,
                error_threshold: float = 1e-5,
                error_check_frequency: int = 5,
                cost_fn: str = 'cosine_distance'):
        """
        :param alpha: Initial value for alpha log scaling stability (n or num_batches x n).
        :param beta: Initial value for beta log scaling stability (m or num_batches x m).
        :param epsilon: Weighting factor of entropy regularization (higher = more entropy, lower = less entropy).
        :param num_iter_max: Maximum number of iterations to perform.
        :param unbalanced_lambda: The weighting factor which controls marginal divergence for unbalanced OT.
        :param error_threshold: Marginal error at which to stop sinkhorn iterations.
        :param error_check_frequency: Frequency with with to check the error.
        :param return_log: Whether to also return a dictionary with logging information.
        :return: The optimal alignment matrix (n x m or num_batches x n x m) according to the cost and marginals
        and optionally a dictionary of logging information.
        """
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.cost_fn = cost_fn
        self.unbalanced_lambda = unbalanced_lambda
        self.error_threshold = error_threshold
        self.error_check_frequency = error_check_frequency

    def forward(self, x, y, a=None, b=None,
                alpha: Optional[torch.FloatTensor] = None,
                beta: Optional[torch.FloatTensor] = None,):
        # Compute cost matrix
        C = self._cost_matrix(x, y, self.cost_fn) 
        
        # Setup
        dtype = C.dtype
        device = C.device
        if len(C.shape) == 2:
            C = C.unsqueeze(0)
        nb, n, m = C.shape
        n_shape = (nb, n)
        m_shape = (nb, m)

        # Use uniform distribution if marginals not provided
        if a is None:
            a = torch.ones(n_shape, dtype=dtype, device=device) / n
        a = a.to(C)

        if b is None:
            b = torch.ones(m_shape, dtype=dtype, device=device) / m
        b = b.to(C)

        # Construct mask (1s for content, 0s for padding)
        mask, mask_n, mask_m = self.compute_masks(C=C, a=a, b=b)
        u_init, v_init = mask_n, mask_m

        # Initialize alpha and beta if not provided (scaling values in log domain to help stability)
        if alpha is None:
            alpha = torch.zeros(n_shape, dtype=dtype, device=device)
        alpha.to(C)

        if beta is None:
            beta = torch.zeros(m_shape, dtype=dtype, device=device)
        beta.to(C)

        # Set u and v
        u_init, v_init = mask_n, mask_m
        u, v = u_init, v_init

        # Check shapes
        assert a.shape == alpha.shape == u.shape == mask_n.shape == n_shape
        assert b.shape == beta.shape == v.shape == mask_m.shape == m_shape
        assert C.shape == mask.shape

        # Define functions to compute K and P
        def compute_log_K(alpha: torch.FloatTensor,
                        beta: torch.FloatTensor) -> torch.FloatTensor:
            """
            Computes log(K) = -C / epsilon.

            :param alpha: Alpha log scaling factor.
            :param beta: Beta log scaling factor.
            :return: log(K) = -C / epsilon with stability and masking.
            """
            return mask * (-C + alpha.unsqueeze(dim=-1) + beta.unsqueeze(dim=-2)) / self.eps

        def compute_K(alpha: torch.FloatTensor,
                    beta: torch.FloatTensor) -> torch.FloatTensor:
            """
            Computes K = exp(-C / epsilon).

            :param alpha: Alpha log scaling factor.
            :param beta: Beta log scaling factor.
            :return: K = exp(-C / epsilon) with stability and masking.
            """
            return torch.exp(compute_log_K(alpha, beta))

        def compute_P(alpha: torch.FloatTensor,
                    beta: torch.FloatTensor,
                    u: torch.FloatTensor,
                    v: torch.FloatTensor) -> torch.FloatTensor:
            """
            Computes transport matrix P = diag(u) * K * diag(v).

            :param alpha: Alpha log scaling factor.
            :param beta: Beta log scaling factor.
            :param u: u vector.
            :param v: v vector.
            :return: P = diag(u) * K * diag(v) with stability and masking.
            """
            return mask * torch.exp(self.mask_log(u, mask_n).unsqueeze(dim=-1) +
                                    compute_log_K(alpha, beta) +
                                    self.mask_log(v, mask_m).unsqueeze(dim=-2))

         # Initialize K
        K = compute_K(alpha, beta)  # nb x n x m
        K_t = K.transpose(-2, -1)  # nb x m x n

        # Set lambda for unbalanced OT
        if self.unbalanced_lambda is not None:
            unbalanced_lambda = self.unbalanced_lambda / (self.unbalanced_lambda + self.eps)
        
        # Set error and num_iter
        error = float('inf')
        num_iter = 0

        # Sinkhorn iterations
        while error > self.error_threshold and num_iter < self.max_iter:
            # Save previous u and v in case of numerical errors
            u_prev, v_prev = u, v

            # Sinkhorn update
            u = (a / (self.bmv(K, v) + 1e-16))
            if self.unbalanced_lambda is not None:
                u **= unbalanced_lambda

            v = (b / (self.bmv(K_t, u) + 1e-16))
            if self.unbalanced_lambda is not None:
                v **= unbalanced_lambda
            
            # Check if we've broken machine precision and if so, return previous result
            unstable = torch.isnan(u).any(dim=-1, keepdim=True) | torch.isnan(v).any(dim=-1, keepdim=True) | \
                    torch.isinf(u).any(dim=-1, keepdim=True) | torch.isinf(v).any(dim=-1, keepdim=True)
            if unstable.any():
                print(f'Warning: Numerical errors at iteration {num_iter}')
                print(f'shape: {m},{n}')
                print(f'epsilon: {self.eps}')
                print(f'error threshold: {self.error_threshold}')
                print(C.max())
                print(C.min())
                pdb.set_trace()
                u = torch.where(unstable.repeat(1, u.size(-1)), u_prev, u)
                v = torch.where(unstable.repeat(1, v.size(-1)), v_prev, v)
                break
            
            # Remove numerical problems in u and v by moving them to K
            alpha += self.eps * self.mask_log(u, mask_n)
            beta += self.eps * self.mask_log(v, mask_m)
            K = compute_K(alpha, beta)
            K_t = K.transpose(-2, -1)

            # Check error
            if num_iter % self.error_check_frequency == 0:
                P = compute_P(alpha, beta, u, v)
                errors = torch.norm(P.sum(dim=-1) - a, dim=-1) ** 2
                error = errors.max()

            # Update num_iter
            num_iter += 1
        
        # Compute optimal transport matrix
        pi = compute_P(alpha, beta, u, v)
            
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    @staticmethod
    def bmv(m: torch.FloatTensor, v: torch.FloatTensor) -> torch.FloatTensor:
        """
        Performs a batched matrix-vector product.

        :param m: A 3-dimensional FloatTensor (num_batches x n1 x n2).
        :param v: A 2-dimensional FloatTensor (num_batches x n2).
        :return: Batched matrix-vector product mv (num_batches x n1).
        """
        assert len(m.shape) == 3
        assert len(v.shape) == 2
        return torch.bmm(m, v.unsqueeze(dim=2)).squeeze(dim=2)

    @staticmethod
    def mask_log(x: torch.FloatTensor, mask: Optional[torch.Tensor] = None) -> torch.FloatTensor:
        """
        Takes the logarithm such that the log of masked entries is zero.

        :param x: FloatTensor whose log will be computed.
        :param mask: Tensor with 1s for content and 0s for padding.
        Entries in x corresponding to 0s will have a log of 0.
        :return: log(x) such that entries where the mask is 0 have a log of 0.
        """
        if mask is not None:
            # Set masked entries of x equal to 1 (in a differentiable way) so log(1) = 0
            mask = mask.float()
            x = x * mask + (1 - mask)

        return torch.log(x)
    
    @staticmethod
    def compute_masks(C: torch.FloatTensor,
                  a: torch.FloatTensor,
                  b: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Computes masks for C, a, and b based on the zero entries in a and b.

        Masks have 1s for content and 0s for padding.

        :param C: Cost matrix (n x m or num_batches x n x m).
        Note: The device and dtype of C are used for all other variables.
        :param a: Row marginals (n or num_batches x n).
        :param b: Column marginals (m or num_batches x m).
        :return: A tuple containing a mask for C, a mask for a, and a mask for b.
        """
        mask_n = (a != 0)
        mask_m = (b != 0)
        mask = mask_n.unsqueeze(dim=-1) & mask_m.unsqueeze(dim=-2)
        mask, mask_n, mask_m = mask.to(C), mask_n.to(C), mask_m.to(C)

        return mask, mask_n, mask_m

    @staticmethod
    def _cost_matrix(x, y, cost_fn: str='cosine_distance'):
        """
        Computes pairwise cost between vectors.

        :param cost_fn: The cost function to use.
        :param x1: A 3D FloatTensor of vectors (bs x n x d).
        :param x2: A 3D FloatTensor of vectors (bs x m x d).
        :return: A 2D FloatTensor of pairwise costs (bs x n x m).
        """
        # Checks on input sizes
        assert len(x.shape) == 3 and len(y.shape) == 3 and x.size(2) == y.size(2)
        
        if cost_fn == 'dot_product':
            C = - torch.bmm(x, y.transpose(1,2))
        elif cost_fn == 'scaled_dot_product':
            cost = - torch.bmm(x, y.transpose(1,2)) / x.size(1)
        elif cost_fn == 'cosine_similarity':
            eps=1e-8
            w1 = x.norm(p=2, dim=2, keepdim=True)  #batch, n, hidden
            w2 = y.norm(p=2, dim=2, keepdim=True)  #batch, m, hidden
            cosine = torch.bmm(x, y.transpose(1,2)) /(w1 * w2.transpose(1,2)).clamp(min=eps)
            cost = - cosine
        elif cost_fn == 'cosine_distance': #
            eps=1e-8
            w1 = x.norm(p=2, dim=2, keepdim=True)  #batch, n, hidden
            w2 = y.norm(p=2, dim=2, keepdim=True)  #batch, m, hidden
            cosine = torch.bmm(x, y.transpose(1,2)) /(w1 * w2.transpose(1,2)).clamp(min=eps)
            cost = 1 - cosine
        else:
            raise ValueError(f'Cost function "{cost_fn}" not supported')
        return cost
