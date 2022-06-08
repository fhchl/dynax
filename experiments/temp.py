### USER CODE

class LoudspeakerDynamics(ControlAffine):
  Bl: float
  Re: float
  Rm: float
  K: float
  L: float
  M: float
  outputs: list = eqx.static_field()

  def __init__(self, params, outputs=[0, 2]):
    self.n_states = 3
    self.n_params = 6
    self.Bl, self.Re, self.Rm, self.K, self.L, self.M = params
    self.outputs = outputs

  def f(self, x, t=None):
    i, d, v = x
    di = (-self.Re*i - self.Bl*v) / self.L
    dd = v
    dv = (self.Bl*i - self.Rm*v - self.K*d) / self.M
    return jnp.array([di, dd, dv])

  def g(self, x, t=None):
    di = 1 / self.L
    dd = 0
    dv = 0
    return jnp.array([di, dd, dv])

  def h(self, x, t=None):
    return x[np.array(self.outputs)]


def main_sastry(sr=96000, n=96000):
  t = jnp.array(np.arange(n)/sr)
  u = jnp.array(np.random.normal(size=n))
  coeffs = dfx.backward_hermite_coefficients(t, u)
  cubic = dfx.CubicInterpolation(t, coeffs)
  ufun = lambda t: cubic.evaluate(t)
  initial_params = [1., 1., 1., 1000., 1e-3, 1e-3]
  dyn = LoudspeakerDynamics(*initial_params)
  true_model = ForwardModel(dyn, sr)
  x0 = jnp.array([0., 0., 0.])
  y = true_model(t, x0, ufun)

  dyn.feedback_linearize(x0)

# main_sastry()

def fit_ml(model: ForwardModel, t, u, y, x0):
  coeffs = dfx.backward_hermite_coefficients(t, u)
  cubic = dfx.CubicInterpolation(t, coeffs)
  ufun = lambda t: cubic.evaluate(t)
  init_params, treedef = jax.tree_flatten(model)
  std_y = np.std(y, axis=0)

  # scale parameters and bounds
  def residuals(params):
    model = treedef.unflatten(params)
    pred_y = model(t, x0, ufun)
    res = ((y - pred_y)/std_y).reshape(-1)
    return res / np.sqrt(len(res))

  # solve least_squares in scaled parameter space
  fun = jax.jit(residuals)
  jac = jax.jit(jax.jacfwd(residuals))
  res = least_squares(fun, init_params, jac=jac, x_scale='jac', verbose=2)
  print(res.x)

def main(n=96000, sr=96000):
  t = jnp.array(np.arange(n)/sr)
  u = jnp.array(np.random.normal(size=n))
  coeffs = dfx.backward_hermite_coefficients(t, u)
  cubic = dfx.CubicInterpolation(t, coeffs)
  ufun = lambda t: cubic.evaluate(t)
  initial_params = [1., 1., 1., 1000., 1e-3, 1e-3]
  dyn = LoudspeakerDynamics(*[i*2 for i in initial_params])
  true_model = ForwardModel(dyn, sr)
  x0 = jnp.array([0., 0., 0.])
  y = true_model(t, x0, ufun)

  model = ForwardModel(LoudspeakerDynamics(*initial_params))
  pred_params = fit_ml(model, t, u, y, x0)
  print(pred_params)

def main_lin():
  initial_params = [1., 1., 1., 1000., 1e-3, 1e-3]
  nonlin = LoudspeakerDynamics(*initial_params)
  lin = nonlin.linearize(jnp.array([0., 0., 0.]))
  print(lin)

  n = 100
  sr = 96000
  t = jnp.array(np.arange(n)/sr)
  u = jnp.array(np.random.normal(size=n))
  coeffs = dfx.backward_hermite_coefficients(t, u)
  cubic = dfx.CubicInterpolation(t, coeffs)
  ufun = lambda t: cubic.evaluate(t)
  x0 = jnp.array([0, 0, 0])
  x = ForwardModel(nonlin)(t, x0, ufun)
  xlin = ForwardModel(lin)(t, x0, ufun)

  print(lin.A, lin.B, lin.C)

  import matplotlib.pyplot as plt
  plt.plot(x)
  plt.plot(xlin, '--')
  plt.show()



  

def generate_feedback_linearizing_law(f, g, h, r):
  # check if feedback linearizable around x0 
  # with collorary 6.17 in Nijmeijer

  # compute poles of linearized system

  # compute poles of feedback linearized system

  # compute feedback gain matrix via pole placement (remark 4.2.3 isidori)
  # and get c0 ... ci vectors

  # compute u wirh remark 4.2.3
  # u = ... + v?
  Lfnh = lie_derivative(f, h, r)
  LgLfn1h = lie_derivative(g, lie_derivative(f, h, r-1))
  return lambda v, x: (-Lfnh(x) + v) / LgLfn1h(x)

class LoudspeakerDynamics1(ControlAffine):
  Bl: float
  Re: float
  Rm: float
  K: float
  L: float
  M: float

  def f(self, x, t=None):
    i, d, v = x
    di = (-self.Re*i - self.Bl*v) / self.L
    dd = v
    dv = (self.Bl*i - self.Rm*v - self.K*d) / self.M
    return jnp.array([di, dd, dv])

  def g(self, x, t=None):
    di = 1 / self.L
    dd = 0
    dv = 0
    return jnp.array([di, dd, dv])

  def h(self, x, t=None):
    return x[1]

def main_2():
  initial_params = [1., 1., 1., 1000., 1e-3, 1e-3]
  dyn = LoudspeakerDynamics1(*initial_params)
  r = 1
  lie = lie_derivative(dyn.f, dyn.h, r)
  print(lie(jnp.array([1., 1., 1.])))

  comp = generate_feedback_linearizing_law(dyn.f, dyn.g, dyn.h, r)
  comp = jax.jit(comp)
  print(comp(-1, jnp.array([0., 1., 0.])))

  x = jnp.array([0.5, 1.5, 0.5])
  for r in range(1, 4):
    print(lie_derivative(dyn.f, dyn.h, r)(x), lie_derivative(dyn.g, lie_derivative(dyn.f, dyn.h, r-1))(x))

