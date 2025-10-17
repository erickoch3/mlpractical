# from mlp.learning_rules import AdamLearningRuleWithWeightDecay
# from mlp.schedulers import CosineAnnealingWithWarmRestarts
import os
from typing import Tuple

import numpy as np

from mlp.layers import DropoutLayer
from mlp.penalties import L1Penalty, L2Penalty


def test_dropout_layer() -> (
    Tuple[bool, np.ndarray, np.ndarray, bool, np.ndarray, np.ndarray]
):
    """Test the DropoutLayer forward and backward propagation.

    This function tests the DropoutLayer implementation by comparing its
    forward propagation (fprop) and backward propagation (bprop) outputs
    against pre-computed correct results.

    Returns:
        Tuple containing:
            - fprop_test (bool): True if fprop output matches expected values.
            - out (ndarray): Actual fprop output from the DropoutLayer.
            - correct_fprop (ndarray): Expected fprop output.
            - bprop_test (bool): True if bprop output matches expected values.
            - grads (ndarray): Actual bprop gradients from the DropoutLayer.
            - correct_bprop (ndarray): Expected bprop gradients.
    """
    # loaded = np.load("../data/correct_results.npz")
    rng = np.random.default_rng(92019)

    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(
        os.path.join(os.environ["MLP_DATA_DIR"], "regularization_debug_pack.npy"),
        allow_pickle=True,
    ).item()

    rng = np.random.default_rng(92019)
    layer = DropoutLayer(rng=rng)

    out = layer.fprop(x)

    grads = layer.bprop(inputs=x, outputs=out, grads_wrt_outputs=np.ones(x.shape))

    #     correct_outputs = correct_outputs['dropout']

    fprop_test = np.allclose(correct_outputs["DropoutLayer_fprop"], out)

    bprop_test = np.allclose(correct_outputs["DropoutLayer_bprop"], grads)

    return (
        fprop_test,
        out,
        correct_outputs["DropoutLayer_fprop"],
        bprop_test,
        grads,
        correct_outputs["DropoutLayer_bprop"],
    )


def test_L1_Penalty():
    """Test the L1Penalty implementation.

    This function tests the L1Penalty class by comparing its penalty value
    calculation (__call__) and gradient computation (grad) against pre-computed
    correct results.

    Returns:
        Tuple containing:
            - call_test (bool): True if penalty value matches expected value.
            - out (float): Actual penalty value computed by L1Penalty.
            - correct_call (float): Expected penalty value.
            - grad_test (bool): True if gradient matches expected gradient.
            - grads (ndarray): Actual gradient computed by L1Penalty.
            - correct_grad (ndarray): Expected gradient.
    """

    rng = np.random.default_rng(92019)

    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(
        os.path.join(os.environ["MLP_DATA_DIR"], "regularization_debug_pack.npy"),
        allow_pickle=True,
    ).item()

    layer = L1Penalty(1e-4)

    out = layer(x)

    grads = layer.grad(x)

    #     correct_outputs = correct_outputs['l1penalty']

    __call__test = np.allclose(correct_outputs["L1Penalty___call__correct"], out)

    grad_test = np.allclose(correct_outputs["L1Penalty_grad_correct"], grads)

    return (
        __call__test,
        out,
        correct_outputs["L1Penalty___call__correct"],
        grad_test,
        grads,
        correct_outputs["L1Penalty_grad_correct"],
    )


def test_L2_Penalty():
    """Test the L2Penalty implementation.

    This function tests the L2Penalty class by comparing its penalty value
    calculation (__call__) and gradient computation (grad) against pre-computed
    correct results.

    Returns:
        Tuple containing:
            - call_test (bool): True if penalty value matches expected value.
            - out (float): Actual penalty value computed by L2Penalty.
            - correct_call (float): Expected penalty value.
            - grad_test (bool): True if gradient matches expected gradient.
            - grads (ndarray): Actual gradient computed by L2Penalty.
            - correct_grad (ndarray): Expected gradient.
    """

    rng = np.random.default_rng(92019)

    x = rng.normal(loc=0, scale=5.0, size=(50, 3, 64, 64))

    correct_outputs = np.load(
        os.path.join(os.environ["MLP_DATA_DIR"], "regularization_debug_pack.npy"),
        allow_pickle=True,
    ).item()

    layer = L2Penalty(1e-4)

    out = layer(x)

    grads = layer.grad(x)

    #     correct_outputs = correct_outputs['l2penalty']

    __call__test = np.allclose(correct_outputs["L2Penalty___call__correct"], out)

    grad_test = np.allclose(correct_outputs["L2Penalty_grad_correct"], grads)

    return (
        __call__test,
        out,
        correct_outputs["L2Penalty___call__correct"],
        grad_test,
        grads,
        correct_outputs["L2Penalty_grad_correct"],
    )
