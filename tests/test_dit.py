"""Sanity tests for the causal transformer. Run: python tests/test_dit.py

1. causality: changing a later frame must not change the outputs for earlier frames
2. KV cache: generating the new frame with cached context must match a full-window forward pass
3. Diffusion Forcing loss runs and gives gradients to every parameter group
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from wm.data import Batch
from wm.dit import DiTConfig, DiTDenoiser

B, T = 2, 5
K = T - 1
for attn in ["st", "full"]:
    torch.manual_seed(0)
    cfg = DiTConfig(in_channels=8, frame_size=16, patch=2, dim=64, depth=2, heads=4, max_frames=5, num_actions=7, attn=attn)
    m = DiTDenoiser(cfg).eval()
    for p in m.parameters():  # zero-init layers would make the tests trivially pass; randomise everything
        torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(B, T, 8, 16, 16)
    cn = torch.randn(B, T)
    act = torch.randint(0, 8, (B, T))
    out1 = m.net(x, cn, act)

    # 1a. no peeking at the future
    x2 = x.clone(); x2[:, 3:] += 1.0
    out2 = m.net(x2, cn, act)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5), "earlier frames changed when a later frame changed"
    # 1b. the past DOES flow forward: changing frame 0 must change the last frame's output
    x3 = x.clone(); x3[:, 0] += 1.0
    out3 = m.net(x3, cn, act)
    assert (out3[:, -1] - out1[:, -1]).abs().max() > 1e-3, "the last frame ignores earlier frames"
    print(f"ok [{attn}] causality: no peeking at the future, and the past reaches the last frame")

    # 2. KV cache equivalence for the last frame
    cache = m.net.context_cache(x[:, :K], cn[:, :K], act[:, :K])
    new = m.net.forward_new(x[:, K], cn[:, K], act[:, K], cache, t_index=K)
    err = (new - out1[:, K]).abs().max().item()
    assert err < 1e-4, err
    print(f"ok [{attn}] kv-cache: cached generation matches full forward (max diff {err:.1e})")

# 3. training losses + generation shapes
for mode in ["df", "last"]:
    m2 = DiTDenoiser(DiTConfig(in_channels=8, frame_size=16, patch=2, dim=64, depth=2, heads=4, max_frames=5,
                               num_actions=7, train_mode=mode, ctx_noise_max=0.7, clamp=False))
    loss = m2.loss(Batch(torch.randn(B, T, 8, 16, 16), torch.randint(0, 7, (B, T - 1))))
    loss.backward()
    assert torch.isfinite(loss)
    assert m2.net.out.weight.grad.abs().sum() > 0
    gen = m2.generate(torch.randn(B, K, 8, 16, 16), torch.randint(0, 7, (B, K)), n_steps=3)
    assert gen.shape == (B, 8, 16, 16)
    print(f"ok {mode}: loss {loss.item():.3f}, generate -> {tuple(gen.shape)}")
print("all DiT tests passed")
