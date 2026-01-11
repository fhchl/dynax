import numpy as np

from dynax.util import multitone


def test_multitone():
    def crest_factor(sig):
        return (
            np.linalg.norm(sig, np.inf) / (np.linalg.norm(sig, 2) / np.sqrt(len(sig)))
        ).item()

    length = 1000
    for num_tones in np.pow(2, np.arange(8)):
        for first_tone in [0, 1, 10]:
            u = multitone(length, num_tones, first_tone)
            cr = crest_factor(u)
            assert cr < 2 or np.allclose(cr, 2)
