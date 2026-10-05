"""CountSketch of full-model gradients for the batch-gradient alignment of Appendix F.2 (Figure 7a).

Each parameter tensor gets its own 2-universal hashes, seeded by the parameter name and a global seed:
bucket h(i) = ((a i + b) mod p) mod m and sign from ((c i + d) mod p) & 1, with p = 2^31 - 1. Inner
products of sketches are unbiased estimates of the exact inner products, with standard deviation about
sqrt((|x|^2 |y|^2 + <x, y>^2) / m). Sketches of the same parameter names with the same seed and m are
comparable across processes and machines, which is what lets the batch-alignment step compare gradients
computed in separate jobs. The paper uses seed 20260923 with m = 2^21.
"""
from __future__ import annotations

import math
import zlib

import torch

PRIME = 2 ** 31 - 1
BATCH_SEED, BATCH_M = 20260923, 1 << 21


def hash_params(name: str, seed: int):
    g = torch.Generator().manual_seed(zlib.crc32(f"{seed}:{name}".encode()))
    a, c = (int(torch.randint(1, PRIME, (1,), generator=g)) for _ in range(2))
    b, d = (int(torch.randint(0, PRIME, (1,), generator=g)) for _ in range(2))
    return a, b, c, d


def sketch_add(sk: torch.Tensor, name: str, x: torch.Tensor, seed: int, offset: int = 0, chunk: int = 1 << 24):
    """Add the CountSketch of the flattened tensor ``x`` into ``sk`` (float64, length m, on its device).

    ``x`` holds the entries of parameter ``name`` starting at flat index ``offset``, so chunked calls
    with offsets give exactly the sketch of the whole parameter.
    """
    m = sk.numel()
    a, b, c, d = hash_params(name, seed)
    flat = x.reshape(-1)
    for s0 in range(0, flat.numel(), chunk):
        v = flat[s0:s0 + chunk].to(sk.device, torch.float64)
        i = torch.arange(offset + s0, offset + s0 + v.numel(), device=sk.device, dtype=torch.int64)
        sgn = 1.0 - 2.0 * (((c * i + d) % PRIME) & 1).to(torch.float64)
        sk.index_add_(0, ((a * i + b) % PRIME) % m, v * sgn)


def selftest():
    """CPU checks: determinism, linearity, chunk invariance, and accuracy on correlated vectors."""
    torch.manual_seed(0)
    dd = torch.float64
    shapes = {"layer.0.w": (1000, 700), "layer.1.w": (300, 2000), "head.w": (1500, 1000), "norm": (4096,)}
    base = {n: torch.randn(s, dtype=dd) for n, s in shapes.items()}
    noise = {n: torch.randn(s, dtype=dd) for n, s in shapes.items()}
    x = base
    y = {n: base[n] + 1.2 * noise[n] for n in shapes}
    m, seed = 1 << 18, 7

    def sk(vec):
        out = torch.zeros(m, dtype=dd)
        for n in shapes:
            sketch_add(out, n, vec[n], seed, chunk=1 << 20)
        return out

    sx, sx2, sy = sk(x), sk(x), sk(y)
    sxy = sk({n: x[n] + y[n] for n in shapes})
    assert torch.equal(sx, sx2), "sketch is not deterministic"
    assert torch.allclose(sx + sy, sxy, atol=1e-8), "sketch is not linear"
    sc = torch.zeros(m, dtype=dd)
    flat = x["layer.0.w"].reshape(-1)
    for s0 in range(0, flat.numel(), 123457):
        sketch_add(sc, "layer.0.w", flat[s0:s0 + 123457], seed, offset=s0, chunk=1 << 20)
    sw = torch.zeros(m, dtype=dd)
    sketch_add(sw, "layer.0.w", x["layer.0.w"], seed, chunk=1 << 20)
    assert torch.allclose(sc, sw, atol=1e-9), "chunked sketch differs from the whole-tensor sketch"
    dot = sum(float((x[n] * y[n]).sum()) for n in shapes)
    nx = math.sqrt(sum(float((x[n] ** 2).sum()) for n in shapes))
    ny = math.sqrt(sum(float((y[n] ** 2).sum()) for n in shapes))
    cos_exact = dot / (nx * ny)
    cos_sk = float(sx @ sy) / float(sx.norm() * sy.norm())
    assert abs(cos_exact - cos_sk) < 0.02, (cos_exact, cos_sk)
    return dict(cos_exact=cos_exact, cos_sketch=cos_sk)


if __name__ == "__main__":
    print("sketch selftest passed:", selftest())
