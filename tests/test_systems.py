import numpy as np
import numpy.testing as npt

from dynax import FeedbackSystem, LinearSystem, SeriesSystem


def test_series():
    n1, m1, p1 = 4, 3, 2
    A1 = np.random.randint(-5, 5, size=(n1, n1))
    B1 = np.random.randint(-5, 5, size=(n1, m1))
    C1 = np.random.randint(-5, 5, size=(p1, n1))
    D1 = np.random.randint(-5, 5, size=(p1, m1))
    sys1 = LinearSystem(A1, B1, C1, D1)
    n2, m2, p2 = 5, p1, 3
    A2 = np.random.randint(-5, 5, size=(n2, n2))
    B2 = np.random.randint(-5, 5, size=(n2, m2))
    C2 = np.random.randint(-5, 5, size=(p2, n2))
    D2 = np.random.randint(-5, 5, size=(p2, m2))
    sys2 = LinearSystem(A2, B2, C2, D2)
    sys = SeriesSystem(sys1, sys2)
    linsys = sys.linearize()
    npt.assert_array_equal(
        linsys.A, np.block([[A1, np.zeros((n1, n2))], [B2.dot(C1), A2]])
    )
    npt.assert_array_equal(linsys.B, np.block([[B1], [B2.dot(D1)]]))
    npt.assert_array_equal(linsys.C, np.block([[D2.dot(C1), C2]]))
    npt.assert_array_equal(linsys.D, D2.dot(D1))


def test_feedback():
    n1, m1, p1 = 4, 3, 2
    A1 = np.random.randint(-5, 5, size=(n1, n1))
    B1 = np.random.randint(-5, 5, size=(n1, m1))
    C1 = np.random.randint(-5, 5, size=(p1, n1))
    D1 = np.zeros((p1, m1))
    sys1 = LinearSystem(A1, B1, C1, D1)
    n2, m2, p2 = 5, p1, 3
    A2 = np.random.randint(-5, 5, size=(n2, n2))
    B2 = np.random.randint(-5, 5, size=(n2, m2))
    C2 = np.random.randint(-5, 5, size=(p2, n2))
    D2 = np.random.randint(-5, 5, size=(p2, m2))
    sys2 = LinearSystem(A2, B2, C2, D2)
    sys = FeedbackSystem(sys1, sys2)
    linsys = sys.linearize()
    npt.assert_array_equal(
        linsys.A, np.block([[A1 + B1 @ D2 @ C1, B1 @ C2], [B2 @ C1, A2]])
    )
    npt.assert_array_equal(linsys.B, np.block([[B1], [np.zeros((n2, m1))]]))
    npt.assert_array_equal(linsys.C, np.block([[C1, np.zeros((p1, n2))]]))
    npt.assert_array_equal(linsys.D, np.zeros((p1, m1)))
