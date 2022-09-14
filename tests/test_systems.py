import numpy as np
from dynax import *


def test_series():
  sys1 = LinearSystem(0, 0, 1, 0)
  sys2 = LinearSystem(0, 0, 1, 0)
  sys = SeriesSystem(sys1, sys2)
  linsys = sys.linearize()
  assert np.allclose(linsys.A, 0)
  assert np.allclose(linsys.B, 0)
  assert np.allclose(linsys.C, 1)
  assert np.allclose(linsys.D, 0)


if __name__ == "__main__":
  tests = [(name, obj)
           for (name, obj) in locals().items()
           if callable(obj) and name.startswith("test_")]
  for name, test in tests:
    test()
