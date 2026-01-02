import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from diffrax import Kvaerno5, PIDController
from jax import Array

from dynax import (
    AbstractSystem,
    field,
    fit_csd_matching,
    fit_least_squares,
    fit_multiple_shooting,
    Flow,
    non_negative_field,
    transfer_function,
)
from dynax.example_models import LotkaVolterra, NonlinearDrag, SpringMassDamper


tols = {"rtol": 1e-02, "atol": 1e-04}


@pytest.mark.parametrize("outputs", [(0,), (0, 1)])
def test_fit_least_squares(outputs):
    # data
    t = np.linspace(0, 1, 100)
    u = (
        0.1 * np.sin(1 * 2 * np.pi * t)
        + np.sin(0.1 * 2 * np.pi * t)
        + np.sin(10 * 2 * np.pi * t)
    )
    true_model = Flow(
        NonlinearDrag(1.0, 2.0, 3.0, 4.0, outputs),
    )
    _, y_true = true_model(t, u)
    # fit
    init_model = Flow(NonlinearDrag(1.0, 2.0, 3.0, 4.0, outputs))
    pred_model = fit_least_squares(init_model, t, y_true, u, verbose=2).result
    # check result
    _, y_pred = pred_model(t, u)
    npt.assert_allclose(y_pred, y_true, **tols)
    assert eqx.tree_equal(pred_model, true_model, **tols)


def test_fit_least_squares_on_batch():
    # data
    t = np.linspace(0, 1, 100)
    us = np.stack(
        (
            np.sin(1 * 2 * np.pi * t),
            np.sin(0.1 * 2 * np.pi * t),
            np.sin(10 * 2 * np.pi * t),
        ),
        axis=0,
    )
    ts = np.repeat(t[None], us.shape[0], axis=0)
    true_model = Flow(
        NonlinearDrag(1.0, 2.0, 3.0, 4.0),
    )
    _, ys = jax.vmap(true_model)(ts, us)
    # fit
    init_model = Flow(
        NonlinearDrag(1.0, 2.0, 3.0, 4.0),
    )
    pred_model = fit_least_squares(init_model, ts, ys, us, batched=True).result
    # check result
    _, ys_pred = jax.vmap(pred_model)(ts, us)
    npt.assert_allclose(ys_pred, ys, **tols)
    assert eqx.tree_equal(pred_model, true_model, **tols)


def test_can_compute_jacfwd_with_implicit_methods():
    # don't get caught by https://github.com/patrick-kidger/diffrax/issues/135
    t = jnp.linspace(0, 1, 10)
    x0 = jnp.array([1.0, 0.0])
    solver_opt = dict(
        solver=Kvaerno5(), stepsize_controller=PIDController(atol=1e-6, rtol=1e-3)
    )

    def fun(m, r, k, x0=x0, solver_opt=solver_opt, t=t):
        model = Flow(SpringMassDamper(m, r, k), **solver_opt)
        x_true, _ = model(t, u=jnp.zeros_like(t), initial_state=x0)
        return x_true

    jac = jax.jacfwd(fun, argnums=(0, 1, 2))
    jac(1.0, 2.0, 3.0)


def test_fit_with_bounded_parameters():
    # data
    t = jnp.linspace(0, 1, 100)
    solver_opt = dict(stepsize_controller=PIDController(rtol=1e-5, atol=1e-7))
    true_model = Flow(
        LotkaVolterra(alpha=2 / 3, beta=4 / 3, gamma=1.0, delta=1.0), **solver_opt
    )
    x_true, _ = true_model(t)
    # fit
    init_model = Flow(
        LotkaVolterra(alpha=1.0, beta=1.0, gamma=1.5, delta=2.0), **solver_opt
    )
    pred_model = fit_least_squares(init_model, t, x_true).result
    # check result
    x_pred, _ = pred_model(t)
    npt.assert_allclose(x_pred, x_true, **tols)
    assert eqx.tree_equal(pred_model, true_model, **tols)


def test_fit_with_bounded_parameters_and_ndarrays():
    # model
    class LotkaVolterraBounded(AbstractSystem):
        alpha: float = field()
        beta: float = field()
        delta_gamma: Array = non_negative_field()

        initial_state = np.array((0.5, 0.5))
        n_inputs = 0

        def vector_field(self, x, u=None, t=None):
            x, y = x
            gamma, delta = self.delta_gamma
            return jnp.array(
                [self.alpha * x - self.beta * x * y, delta * x * y - gamma * y]
            )

    # data
    t = jnp.linspace(0, 1, 100)
    solver_opt = dict(stepsize_controller=PIDController(rtol=1e-5, atol=1e-7))
    true_model = Flow(
        LotkaVolterraBounded(
            alpha=2 / 3, beta=4 / 3, delta_gamma=jnp.array([1.0, 1.0])
        ),
        **solver_opt,
    )
    x_true, _ = true_model(t)
    # fit
    init_model = Flow(
        LotkaVolterraBounded(alpha=1.0, beta=1.0, delta_gamma=jnp.array([1.5, 2])),
        **solver_opt,
    )
    pred_model = fit_least_squares(init_model, t, x_true).result
    # check result
    x_pred, _ = pred_model(t)
    assert eqx.tree_equal(pred_model, true_model, **tols)
    npt.assert_allclose(x_pred, x_true, **tols)


@pytest.mark.parametrize("num_shots", [1, 2, 3])
def test_fit_multiple_shooting_with_input(num_shots):
    # data
    t = jnp.linspace(0, 1, 200)
    u = jnp.sin(1 * 2 * np.pi * t)
    true_model = Flow(SpringMassDamper(1.0, 2.0, 3.0))
    x_true, _ = true_model(t, u)
    # fit
    init_model = Flow(SpringMassDamper(1.0, 1.0, 1.0))
    pred_model = fit_multiple_shooting(
        init_model,
        t,
        x_true,
        u,
        continuity_penalty=1,
        num_shots=num_shots,
        verbose=2,
    ).result
    # check result
    x_pred, _ = pred_model(t, u)
    npt.assert_allclose(x_pred, x_true, **tols)
    assert eqx.tree_equal(pred_model, true_model, **tols)


@pytest.mark.parametrize("num_shots", [1, 2, 3])
def test_fit_multiple_shooting_without_input(num_shots):
    # data
    t = jnp.linspace(0, 1, 200)
    solver_opt = dict(stepsize_controller=PIDController(rtol=1e-3, atol=1e-6))
    true_model = Flow(
        LotkaVolterra(alpha=2 / 3, beta=4 / 3, gamma=1.0, delta=1.0), **solver_opt
    )
    x_true, _ = true_model(t)
    # fit
    init_model = Flow(
        LotkaVolterra(alpha=1.0, beta=1.0, gamma=1.5, delta=2.0), **solver_opt
    )
    pred_model = fit_multiple_shooting(
        init_model, t, x_true, num_shots=num_shots, continuity_penalty=1
    ).result
    # check result
    x_pred, _ = pred_model(t)
    npt.assert_allclose(x_pred, x_true, atol=1e-3, rtol=1e-3)
    assert eqx.tree_equal(
        pred_model,
        true_model,
        atol=1e-2,
        rtol=1e-2,
    )


def test_transfer_function():
    sys = SpringMassDamper(1.0, 1.0, 1.0)
    sr = 100
    f = jnp.linspace(0, sr / 2, 100)
    s = 2 * np.pi * f * 1j
    H = jax.vmap(transfer_function(sys))(s)[:, 0]
    H_true = 1 / (sys.m * s**2 + sys.r * s + sys.k)
    npt.assert_array_almost_equal(H, H_true)


def test_csd_matching():
    from scipy.signal import butter, lfilter
    from scipy.signal.windows import tukey

    np.random.seed(123)
    # model
    sys = SpringMassDamper(1.0, 20.0, 1.0)
    model = Flow(sys, stepsize_controller=PIDController(rtol=1e-5, atol=1e-5))
    # input is lowpassed and windowed noise
    duration = 100
    sr = 40
    t = np.arange(int(duration * sr)) / sr
    b, a = butter(4, sr / 2 * 0.7, fs=sr)
    u = lfilter(b, a, np.random.normal(size=len(t))) * tukey(len(t))
    # output
    _, y = model(t, u)
    # fit
    init_sys = sys
    nperseg = sr * 5
    fitted_sys = fit_csd_matching(init_sys, u, y, sr, nperseg=nperseg, verbose=2).result

    assert eqx.tree_equal(
        fitted_sys,
        sys,
        rtol=1e-1,  # yep, this method is not very accurate
        atol=1e-1,
    )


def test_estimate_initial_state():
    class NonlinearDragFreeInitialState(NonlinearDrag):
        initial_state: Array = field(init=False)

        def __post_init__(self):
            self.initial_state = jnp.zeros(2)

    # data
    t = np.linspace(0, 2, 200)
    u = (
        np.sin(1 * 2 * np.pi * t)
        + np.sin(0.1 * 2 * np.pi * t)
        + np.sin(10 * 2 * np.pi * t)
    )

    # True model has nonzero initial state
    true_initial_state = jnp.array([1.0, 0.5])
    true_model = Flow(NonlinearDragFreeInitialState(1.0, 2.0, 3.0, 4.0, outputs=(0, 1)))
    true_model = eqx.tree_at(
        lambda t: t.system.initial_state, true_model, true_initial_state
    )
    _, y_true = true_model(t, u, true_initial_state)

    # fit
    init_model = Flow(NonlinearDragFreeInitialState(1.0, 1.0, 1.0, 1.0, outputs=(0, 1)))
    pred_model = fit_least_squares(init_model, t, y_true, u=u).result

    # check result
    _, y_pred = pred_model(t, u)
    npt.assert_allclose(y_pred, y_true, **tols)
    npt.assert_allclose(
        pred_model.system.initial_state,
        true_initial_state,
        **tols,
    )
