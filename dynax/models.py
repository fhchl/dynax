from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from equinox import static_field
from jax.tree_util import tree_flatten

from .estimation import boxed_field, non_negative_field
from .system import ControlAffine, DynamicalSystem


class SpringMassDamper(DynamicalSystem):
  """Forced second-order linear spring-mass-damper system.

  .. math:: m x'' + r x' + k x = u.

  """
  m: float
  r: float
  k: float
  n_states = 2
  n_inputs = 1
  n_params = 3
  def vector_field(self, x, u=None, t=None):
    u = u.squeeze() if u is not None else 0
    x1, x2 = x
    return jnp.array([x2, (u - self.r*x2 - self.k*x1)/self.m])


class NonlinearDrag(ControlAffine):
  """Spring-mass-damper system with nonlin drag.

  .. math:: m x'' +  r x' + r2 x'|x'| + k x = u.

  """
  m: float
  r: float
  r2: float
  k: float
  n_states = 2
  n_inputs = 1
  n_outputs = 1
  n_params = 4
  def f(self, x, u=None, t=None):
    x1, x2 = x
    return jnp.array(
      [x2, (- self.r * x2 - self.r2 * jnp.abs(x2) + x2 - self.k * x1)/self.m])
  def g(self, x, u=None, t=None):
    return jnp.array([0., 1./self.m])
  def h(self, x, u=None, t=None):
    return x[0]


class Sastry9_9(ControlAffine):
  """Sastry Example 9.9"""
  n_states = 3
  n_inputs = 1
  n_params = 0
  def f(self, x, t=None): return jnp.array([0., x[0] + x[1]**2, x[0] - x[1]])
  def g(self, x, t=None): return jnp.array([jnp.exp(x[1]), jnp.exp(x[1]), 0.])
  def h(self, x, t=None): return x[2]


class LotkaVolterra(DynamicalSystem):
  alpha: float = non_negative_field()
  beta: float = non_negative_field()
  gamma: float = non_negative_field()
  delta: float = non_negative_field()

  def vector_field(self, x, u=None, t=None):
    x, y = x
    return jnp.array([self.alpha * x - self.beta * x * y,
                      self.delta * x * y - self.gamma * y])


class SpringMassWithBoucWenHysteresis(DynamicalSystem):
  """https://en.wikipedia.org/wiki/Bouc%E2%80%93Wen_model_of_hysteresis"""
  m: float = non_negative_field()            # kg
  r: float = non_negative_field()            # Ns/m
  ki: float = non_negative_field()           # N/m
  gamma: float = non_negative_field()
  n: float = non_negative_field(min_val=1.)
  a: float = boxed_field(0., 1.)
  def vector_field(self, x, u=None, t=None):
    if u is None: u = 0
    f = u
    u, du, z = x
    # remove parameter redundancies
    A = 1
    beta = A - self.gamma
    # restoring force with hysteresis
    F = self.a*self.ki*u + (1 - self.a)*self.ki*z
    # shape control function
    psi = beta * jnp.sign(z*du) + self.gamma
    return jnp.array([du,
                      (f - self.r*du - F)/self.m,
                      du * (A - psi * jnp.power(jnp.abs(z), self.n))])

from dataclasses import field

def fac(*args, **kwargs):
  2+2
  pass

class PolyNonLinSLSL2R2GenCunLiDyn(ControlAffine):
  """The full model."""
  n_inputs = 1
  n_states = 5
  Bl: float = non_negative_field()
  Re: float = non_negative_field()
  Rm: float = non_negative_field()
  K: float = non_negative_field()
  L: float = non_negative_field()
  M: float = non_negative_field()
  L20: float = non_negative_field()
  R20: float = non_negative_field()
  K2: float = non_negative_field()
  K2divR2: float = non_negative_field()
  Bln: List[float]
  Kn: List[float]
  Ln: List[float]
  Li: List[float]
  out: List[int] = static_field(default_factory=lambda: [1])

  def __init__(self, out=[1]):
    self.Bl = 3.293860219666026
    self.Re = 6.951042909533158
    self.Rm = 0.7237227039672062
    self.K = 1927.6900850359816
    self.L = 3.0198137447786782e-05
    self.M = 0.0026736262193096066
    self.L20 = 0.0002482116897900503
    self.R20 = 2.5460367443216683
    self.K2 = 205.3268589910077
    self.K2divR2 = 192.45813523048824
    self.Bln = [-1534976847.641551, 3086545.3414503504, -103586.95986556074, 53.36132745159803]
    self.Kn = [3226388093843.6885, -431643235.42306566, -19460708.731581513, -16921.87605383064]
    self.Ln = [29040.803548406657, -194.6354429605707, 0.39969456055741825, -0.0003743634072813242]
    self.Li = [0.000995183661999823, 0.0010163108959166816]
    self.n_params = 3*4 + 2 + 10
    self.out = out
    self.n_outputs = len(out)

  def __post_init__(self):
    super().__init__()

  def tf(self, f):
    tree, treedef = tree_flatten(self)
    Bl, Re, Rm, K, L, M, L2, R2, K2, K2divR2 = tree[:10]
    Rm2 = K2/K2divR2
    s = 1j*2*jnp.pi*f
    sZm = s**2*M + Rm*s + K + s*Rm2*K2/(K2 + s*Rm2)  # derivative of mech. impedance
    Ze = Re + s*L + R2*L2*s/(R2+L2*s)  # electrical impedance
    D = Bl / (sZm*Ze + Bl**2*s)
    V = s*D
    I = (1 - Bl*s*D) / Ze
    return jnp.stack((I, D, V), axis=-1)

  def _Bl(self, d):
    """Displacement dependend force-factor."""
    return jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)

  def _K(self, d):
    """Displacement dependend stiffness."""
    return jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)

  def _L(self, d, i):
    """Displacement and current dependend inductance."""
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.)  # L(i) = poly + 1
    return jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) #  L = L(d)L(i)

  def f(self, x, t=None):
    i, d, v, i2, d2 = jnp.moveaxis(x, -1, 0)
    Bl = self._Bl(d)
    Re = self.Re
    Rm = self.Rm
    K = self._K(d)
    M = self.M
    L = self._L(d, i)
    L_d, L_i = jax.grad(self._L, argnums=(0, 1))(d, i)  # ∂L/∂d, ∂L/∂i
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 =  self.L  # L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    L2_i = L20 * L_i/L0
    # standard linear solid
    # NOTE: how to scale these parameters? Like L2R2?
    K2 = self.K2
    K2divR2 = self.K2divR2  # K₂/R₂
    # state evolution
    di = (-(Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / (L + i*L_i)
    dd = v
    dv = ((Bl + 0.5*(L_d*i + L2_d*i2))*i - Rm*v - K*d - K2*d2) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / (L2 + i*L2_i)
    dd2 = v - K2divR2*d2   # ẋ₂  = ẋ - K₂/R₂ ẋ
    return jnp.array([di, dd, dv, di2, dd2])

  def g(self, x, t=None):
    i, d, _, _, _ = x
    L = self._L(d, i)
    L_i = jax.grad(self._L, argnums=1)(d, i)  # ∂L/∂i  # TODO: L_i is computed twice
    return jnp.array([1/(L + i*L_i), 0., 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array(self.out)]


class PolyNonLinL2R2GenCunDyn(ControlAffine):
  """The full model."""
  n_inputs = 1
  n_states = 4
  Bl: float = non_negative_field()
  Re: float = non_negative_field()
  Rm: float = non_negative_field()
  K: float = non_negative_field()
  L: float = non_negative_field()
  M: float = non_negative_field()
  L20: float = non_negative_field()
  R20: float = non_negative_field()
  Bln: List[float]
  Kn: List[float]
  Ln: List[float]
  out: List[int] = static_field(default_factory=lambda: [1])

  def __init__(self, out=[1]):
    self.Bl = 3.293860219666026
    self.Re = 6.951042909533158
    self.Rm = 0.7237227039672062
    self.K = 1927.6900850359816
    self.L = 3.0198137447786782e-05
    self.M = 0.0026736262193096066
    self.L20 = 0.0002482116897900503
    self.R20 = 2.5460367443216683
    self.Bln = [-1534976847.641551, 3086545.3414503504, -103586.95986556074, 53.36132745159803]
    self.Kn = [3226388093843.6885, -431643235.42306566, -19460708.731581513, -16921.87605383064]
    self.Ln = [29040.803548406657, -194.6354429605707, 0.39969456055741825, -0.0003743634072813242]
    self.out = out
    self.n_outputs = len(out)

  def __post_init__(self):
    super().__init__()

  def tf(self, f):
    tree, treedef = tree_flatten(self)
    Bl, Re, Rm, K, L, M, L2, R2 = tree[:8]
    s = 1j*2*jnp.pi*f
    sZm = s**2*M + Rm*s + K   # derivative of mech. impedance
    Ze = Re + s*L + R2*L2*s/(R2+L2*s)  # electrical impedance
    D = Bl / (sZm*Ze + Bl**2*s)
    V = s*D
    I = (1 - Bl*s*D) / Ze
    return jnp.stack((I, D, V), axis=-1)

  def _Bl(self, d):
    """Displacement dependend force-factor."""
    return jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)

  def _K(self, d):
    """Displacement dependend stiffness."""
    return jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)

  def _L(self, d, i):
    """Displacement and current dependend inductance."""
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    return jnp.polyval(Ld_coefs, d)

  def f(self, x, t=None):
    i, d, v, i2 = jnp.moveaxis(x, -1, 0)
    Bl = self._Bl(d)
    Re = self.Re
    Rm = self.Rm
    K = self._K(d)
    M = self.M
    L = self._L(d, i)
    L_d, L_i = jax.grad(self._L, argnums=(0, 1))(d, i)  # ∂L/∂d, ∂L/∂i
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 =  self.L  # L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    L2_i = L20 * L_i/L0
    # state evolution
    di = (-(Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / (L + i*L_i)
    dd = v
    dv = ((Bl + 0.5*(L_d*i + L2_d*i2))*i - Rm*v - K*d) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / (L2 + i*L2_i)
    return jnp.array([di, dd, dv, di2])

  def g(self, x, t=None):
    i, d, _, _ = x
    L = self._L(d, i)
    L_i = jax.grad(self._L, argnums=1)(d, i)  # ∂L/∂i  # TODO: L_i is computed twice
    return jnp.array([1/(L + i*L_i), 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array(self.out)]



class PolyNonLinLS(ControlAffine):
  n_inputs = 1
  n_states = 3
  Bl: float = non_negative_field(default=3.293860219666026)
  Re: float = non_negative_field(default=6.951042909533158)
  Rm: float = non_negative_field(default=0.7237227039672062)
  K: float = non_negative_field(default=1927.6900850359816)
  L: float = non_negative_field(default=3.0198137447786782e-05)
  M: float = non_negative_field(default=0.0026736262193096066)
  Bln: List[float]
  Kn: List[float]
  Ln: List[float]
  out: List[int] = static_field(default_factory=lambda: [1])

  def __init__(self, out=[1]):
    self.Bln = [-1534976847.641551, 3086545.3414503504, -103586.95986556074, 53.36132745159803]
    self.Kn = [3226388093843.6885, -431643235.42306566, -19460708.731581513, -16921.87605383064]
    self.Ln = [29040.803548406657, -194.6354429605707, 0.39969456055741825, -0.0003743634072813242]
    self.out = out
    self.n_outputs = len(out)

  def _Bl(self, d):
    """Displacement dependend force-factor."""
    return jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)

  def _K(self, d):
    """Displacement dependend stiffness."""
    return jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)

  def _L(self, d):
    """Displacement and current dependend inductance."""
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    return jnp.polyval(Ld_coefs, d)

  def f(self, x, t=None):
    i, d, v = jnp.moveaxis(x, -1, 0)
    Bl = self._Bl(d)
    Re = self.Re
    Rm = self.Rm
    K = self._K(d)
    M = self.M
    L = self._L(d)
    L_d = jax.grad(self._L)(d)
    # state evolution
    di = (-(Re)*i - (Bl + L_d*i)*v) / (L)
    dd = v
    dv = ((Bl + 0.5*(L_d*i))*i - Rm*v - K*d) / M
    return jnp.array([di, dd, dv])

  def g(self, x, t=None):
    i, d, _ = x
    L = self._L(d)
    return jnp.array([1/(L), 0., 0.])

  def h(self, x, t=None):
    return x[np.array(self.out)]