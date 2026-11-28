import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from omlx.patches.deepseek_v41_dfly2.dfly2_heads import final_norm, hidden_correction, mtp_fusion

EPS = 1e-6


def _eval(x):
    mx.eval(x)
    return np.asarray(x)


def _rms(x, weight):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * weight


def _linear(x, weight):
    return x @ weight.T


def test_mtp_fusion_matches_numpy_reference():
    anchor = np.array([[[1.0, -2.0]]], np.float32)
    target = np.array([[[0.5, 3.0]]], np.float32)
    w = {
        "model.mtp_fusion.enorm.weight": mx.array([2.0, 0.5]),
        "model.mtp_fusion.hnorm.weight": mx.array([1.5, 1.0]),
        "model.mtp_fusion.eh_proj.weight": mx.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, -0.5]]),
        "model.mtp_fusion.gate": mx.array([0.25, -0.5]),
    }
    result = _eval(mtp_fusion(mx.array(anchor), mx.array(target), w, eps=EPS))
    fused = _linear(np.concatenate([_rms(anchor, np.array([2.0, 0.5])), _rms(target, np.array([1.5, 1.0]))], -1), np.array([[1, 0, .5, 0], [0, 1, 0, -.5]], np.float32))
    expected = anchor + np.array([.25, -.5], np.float32) * fused
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_hidden_correction_and_final_norm_match_numpy_reference():
    hidden = np.array([[1.0, -1.0]], np.float32)
    previous = np.array([[2.0, 1.0]], np.float32)
    w = {
        "model.hidden_correction.hidden_norm.weight": mx.array([1.0, 2.0]),
        "model.hidden_correction.embed_norm.weight": mx.array([0.5, 1.5]),
        "model.hidden_correction.gate_proj.weight": mx.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]),
        "model.hidden_correction.up_proj.weight": mx.array([[0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]]),
        "model.hidden_correction.down_proj.weight": mx.array([[1.0, 0.0], [0.0, -1.0]]),
        "model.final_norm.weight": mx.array([1.25, 0.75]),
    }
    result = _eval(hidden_correction(mx.array(hidden), mx.array(previous), w, eps=EPS))
    z = np.concatenate([_rms(hidden, np.array([1.0, 2.0])), _rms(previous, np.array([.5, 1.5]))], -1)
    gate = _linear(z, np.array([[1, 0, 0, 1], [0, 1, 1, 0]], np.float32))
    silu = gate / (1 + np.exp(-gate))
    up = _linear(z, np.array([[.5, 0, 1, 0], [0, .5, 0, 1]], np.float32))
    expected = hidden + _linear(silu * up, np.array([[1, 0], [0, -1]], np.float32))
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_eval(final_norm(mx.array(result), w, eps=EPS)), _rms(expected, np.array([1.25, .75])), rtol=1e-6, atol=1e-6)
