"""
Optional PyTorch/GPU batched classical ER baseline.

PyTorch is an optional dependency; import this module only when it is
available.  On CPU it uses float64 for a fair comparison with NumPy; on CUDA
it uses float32 for throughput.
"""

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def er_combine_torch(w: np.ndarray, betas: np.ndarray,
                     device: str | None = None):
    """
    Batched ER combination on a PyTorch device.

    Returns NumPy arrays for direct comparison with the reference baseline.
    """
    if torch is None:
        raise ImportError("PyTorch is not installed")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if device == "cpu" else torch.float32

    w_t = torch.as_tensor(w, dtype=dtype, device=device)
    betas_t = torch.as_tensor(betas, dtype=dtype, device=device)
    B, L, N = betas_t.shape
    if tuple(w_t.shape) != (B, L):
        raise ValueError("w must have shape (batch, L)")

    m = torch.zeros((B, L, N + 2), dtype=dtype, device=device)
    active = w_t > 1e-12
    m[..., :N] = w_t[..., None] * betas_t
    m[..., N] = 1.0 - w_t
    m[..., N + 1] = w_t * torch.clamp(1.0 - betas_t.sum(dim=2), min=0.0)
    m[~active, N] = 1.0

    m_c = m[:, 0, :].clone()
    for k in range(1, L):
        m_n = m[:, k, :]
        mc_N = m_c[:, :N]
        mn_N = m_n[:, :N]
        conflict = mc_N.sum(dim=1) * mn_N.sum(dim=1) - (mc_N * mn_N).sum(dim=1)
        K = 1.0 / (1.0 - conflict + 1e-15)
        mD_c = m_c[:, N] + m_c[:, N + 1]
        mD_n = m_n[:, N] + m_n[:, N + 1]

        m_new = torch.empty_like(m_c)
        m_new[:, :N] = K[:, None] * (
            mc_N * mn_N + mc_N * mD_n[:, None] + mD_c[:, None] * mn_N
        )
        m_new[:, N + 1] = K * (
            m_c[:, N + 1] * m_n[:, N + 1]
            + m_c[:, N + 1] * m_n[:, N]
            + m_c[:, N] * m_n[:, N + 1]
        )
        m_new[:, N] = K * m_c[:, N] * m_n[:, N]
        m_c = m_new

    denom = 1.0 - m_c[:, N] + 1e-15
    beta = m_c[:, :N] / denom[:, None]
    beta_D = m_c[:, N + 1] / denom
    return beta.cpu().numpy(), beta_D.cpu().numpy()
