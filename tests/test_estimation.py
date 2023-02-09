import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from diffrax import Kvaerno5, PIDController
from dynax import fit_least_squares, Flow
from dynax.estimation import fit_csd_matching, fit_multiple_shooting, transfer_function
from dynax.example_models import LotkaVolterra, SpringMassDamper


tols = dict(rtol=1e-05, atol=1e-08)


def test_fit_least_squares():
    # data
    t = np.linspace(0, 1, 100)
    u = np.sin(1 * 2 * np.pi * t)
    x0 = [1.0, 0.0]
    true_model = Flow(SpringMassDamper(1.0, 2.0, 3.0))
    x_true, _ = true_model(x0, t, u)
    # fit
    init_model = Flow(SpringMassDamper(1.0, 1.0, 1.0))
    pred_model = fit_least_squares(init_model, t, x_true, x0, u)
    # check result
    x_pred, _ = pred_model(x0, t, u)
    npt.assert_allclose(x_pred, x_true, **tols)
    npt.assert_allclose(
        jax.tree_util.tree_flatten(pred_model)[0],
        jax.tree_util.tree_flatten(true_model)[0],
        **tols
    )


def test_can_compute_jacfwd_with_implicit_methods():
    # don't get catched by https://github.com/patrick-kidger/diffrax/issues/135
    t = jnp.linspace(0, 1, 10)
    x0 = jnp.array([1.0, 0.0])
    solver_opt = dict(solver=Kvaerno5(), step=PIDController(atol=1e-6, rtol=1e-3))

    def fun(m, r, k, x0=x0, solver_opt=solver_opt, t=t):
        model = Flow(SpringMassDamper(m, r, k), **solver_opt)
        x_true, _ = model(x0, t)
        return x_true

    jac = jax.jacfwd(fun, argnums=(0, 1, 2))
    jac(1.0, 2.0, 3.0)


def test_fit_with_bouded_parameters():
    # data
    t = np.linspace(0, 1, 100)
    x0 = [0.5, 0.5]
    solver_opt = dict(step=PIDController(rtol=1e-5, atol=1e-7))
    true_model = Flow(
        LotkaVolterra(alpha=2 / 3, beta=4 / 3, gamma=1.0, delta=1.0), **solver_opt
    )
    x_true, _ = true_model(x0, t)
    # fit
    init_model = Flow(
        LotkaVolterra(alpha=1.0, beta=1.0, gamma=1.5, delta=2.0), **solver_opt
    )
    pred_model = fit_least_squares(init_model, t, x_true, x0)
    # check result
    x_pred, _ = pred_model(x0, t)
    npt.assert_allclose(x_pred, x_true, **tols)
    npt.assert_allclose(
        jax.tree_util.tree_flatten(pred_model)[0],
        jax.tree_util.tree_flatten(true_model)[0],
        **tols
    )


def test_fit_multiple_shooting_with_input():
    # data
    t = np.linspace(0, 1, 100)
    u = np.sin(1 * 2 * np.pi * t)
    x0 = [1.0, 0.0]
    true_model = Flow(SpringMassDamper(1.0, 2.0, 3.0))
    x_true, _ = true_model(x0, t, u)
    # fit
    init_model = Flow(SpringMassDamper(1.0, 1.0, 1.0))
    pred_model = fit_multiple_shooting(
        init_model, t, x_true, x0, u, continuity_penalty=2
    )[0]
    # check result
    x_pred, _ = pred_model(x0, t, u)
    npt.assert_allclose(x_pred, x_true, **tols)
    npt.assert_allclose(
        jax.tree_util.tree_flatten(pred_model)[0],
        jax.tree_util.tree_flatten(true_model)[0],
        **tols
    )


def test_fit_multiple_shooting_without_input():
    tols = dict(rtol=1e-04, atol=1e-4)
    # data
    t = np.linspace(0, 1, 100)
    x0 = [0.5, 0.5]
    solver_opt = dict(step=PIDController(rtol=1e-5, atol=1e-7))
    true_model = Flow(
        LotkaVolterra(alpha=2 / 3, beta=4 / 3, gamma=1.0, delta=1.0), **solver_opt
    )
    x_true, _ = true_model(x0, t)
    # fit
    init_model = Flow(
        LotkaVolterra(alpha=1.0, beta=1.0, gamma=1.5, delta=2.0), **solver_opt
    )
    pred_model = fit_multiple_shooting(
        init_model, t, x_true, x0, num_shots=3, continuity_penalty=2
    )[0]
    # check result
    x_pred, _ = pred_model(x0, t)
    npt.assert_allclose(x_pred, x_true, **tols)
    npt.assert_allclose(
        jax.tree_util.tree_flatten(pred_model)[0],
        jax.tree_util.tree_flatten(true_model)[0],
        **tols
    )


def test_transfer_function():
    sys = SpringMassDamper(1.0, 1.0, 1.0)
    sr = 100
    f = np.linspace(0, sr / 2, 100)
    s = 2 * np.pi * f * 1j
    H = jax.vmap(transfer_function(sys))(s)[:, 0, 0]
    H_true = 1 / (sys.m * s**2 + sys.r * s + sys.k)
    npt.assert_array_almost_equal(H, H_true)


def test_csd_matching():
    np.random.seed(123)
    # model
    sys = SpringMassDamper(1.0, 1.0, 1.0)
    model = Flow(sys, step=PIDController(rtol=1e-4, atol=1e-6))
    x0 = np.zeros(sys.n_states)
    # input
    duration = 1000
    sr = 50
    t = np.arange(int(duration * sr)) / sr
    u = np.random.normal(size=len(t))
    # output
    _, y = model(x0, t, u)
    # fit
    init_sys = SpringMassDamper(1.0, 1.0, 1.0)
    fitted_sys = fit_csd_matching(init_sys, u, y, sr, nperseg=1024, verbose=1)

    npt.assert_allclose(
        jax.tree_util.tree_flatten(fitted_sys)[0],
        jax.tree_util.tree_flatten(sys)[0],
        rtol=1e-1,
        atol=1e-1,
    )

    # import matplotlib.pyplot as plt
    # fitted_model = ForwardModel(fitted_sys, step=PIDController(rtol=1e-4, atol=1e-6))
    # _, y_pred = fitted_model(x0, t, u)

    # plt.figure()
    # plt.plot(t, y, 'k-')
    # plt.plot(t, y_pred, 'b--')

    # plt.figure()
    # H = transfer_function(sys)
    # f = np.linspace(0, sr/2, 1000)
    # s = 2*np.pi*f*1j
    # h = jax.vmap(H)(s)
    # m, r, k = sys.m, sys.r, sys.k
    # plt.plot(f, np.abs(h)[:, 0, 0], label='true')
    # plt.plot(f, np.abs(1/(m*s**2 + r*s + k)), label='theory')

    # fitted_H = transfer_function(fitted_sys)
    # plt.plot(f, np.abs(jax.vmap(fitted_H)(s))[:, 0, 0], label='estimate')
    # plt.legend()
    # plt.show()
