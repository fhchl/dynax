import dynax as dx
import numpy as np
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import jax

class LoudspeakerDynamics(dx.ControlAffine):
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

class LoudspeakerDynamics2(LoudspeakerDynamics):
  Bl1: float

  def __init__(self, params, outputs=[0, 2]):
    self.n_states = 3
    self.n_params = 7
    self.Bl1, self.Bl, self.Re, self.Rm, self.K, self.L, self.M = params
    self.outputs = outputs

  def f(self, x, t=None):
    i, d, v = x
    Bl = (self.Bl1*d + self.Bl)
    di = (-self.Re*i - Bl*v) / self.L
    dd = v
    dv = (Bl*i - self.Rm*v - self.K*d) / self.M
    return jnp.array([di, dd, dv])

sr = 96000
n = 96000
t = jnp.array(np.arange(n)/sr)
u = jnp.array(np.random.normal(size=n))
coeffs = dfx.backward_hermite_coefficients(t, u)
cubic = dfx.CubicInterpolation(t, coeffs)
ufun = lambda t: cubic.evaluate(t)
initial_params = [31., 2., 3., 5., 7., 11., 13.]
initial_params = [2., 3., 5., 7., 11., 13.]
x0 = jnp.array([17., 19., 23.])
u = 29

Os = []
outs = [
  # [0, 1, 2], [0, 1], [0, 2], 
  # [0, 2], 
  [0],
  [1],
  [2]
]
for out in outs:
  dyn = LoudspeakerDynamics(initial_params, outputs=out)
  O = dyn.obs_ident_mat(x0, u)
  Os.append(O)
  print("Out:", out)
  print("Rank:", np.linalg.matrix_rank(O))


def unidentifiable_params(O, n_states, names):
  """Compute unidentifiable parameters iteratively."""
  assert O.shape[1] == len(names)
  res = {}
  unidentifiable = []
  for i in range(n_states, O.shape[1]):
    Onew = np.delete(O, i, axis=1)
    if np.linalg.matrix_rank(Onew) == np.linalg.matrix_rank(O):
      unidentifiable.append(i)
  
  if unidentifiable:
    for uni in unidentifiable:
      Onew = np.delete(O, uni, axis=1)
      namesnew = np.delete(names, uni)
      res[names[uni]] = unidentifiable_params(Onew, n_states, namesnew)
  return res

from PrettyPrint import PrettyPrintTree
pt = PrettyPrintTree(default_orientation=PrettyPrintTree.HORIZONTAL)
n_states = 3
names = ['i', 'd', 'v', 'Bl1', 'Bl', 'Re', 'Rm', 'K', 'L', 'M']
names = ['i', 'd', 'v',  'Bl', 'Re', 'Rm', 'K', 'L', 'M']

for out, O in zip(outs, Os):
  pt.print_json(unidentifiable_params(O, n_states, names), name=out)
  print()




