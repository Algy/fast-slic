import numpy as np
import pytest

from fast_slic.crf import SimpleCRF

def test_crf_basic():
    crf = SimpleCRF(3, 100)
    assert crf.space_size == 300
    assert crf.first_time == -1
    assert crf.last_time == -1
    assert crf.num_frames == 0
    with pytest.raises(IndexError):
        crf.get_frame(10)
    assert crf.pop_frame() == -1


def test_crf_frame():
    crf = SimpleCRF(3, 100)
    frame = crf.push_frame()
    assert crf.num_frames == 1
    assert crf.first_time == frame.time
    assert crf.last_time == frame.time
    assert frame.space_size == 300
    assert frame.time == 0
    assert crf.get_frame(0).time == 0


def test_crf_frame_2():
    crf = SimpleCRF(3, 100)
    frame_1 = crf.push_frame()
    frame_2 = crf.push_frame()

    assert crf.num_frames == 2
    assert crf.first_time == frame_1.time
    assert crf.last_time == frame_2.time

    assert crf.pop_frame() == 0
    assert crf.first_time == crf.last_time == 1

def test_gc():
    crf = SimpleCRF(3, 100)
    frame = crf.push_frame()
    del crf
    
    # frame should be alive
    frame.unaries

def test_unaries():
    crf = SimpleCRF(3, 3)
    frame = crf.push_frame()

    frame.set_unbiased()
    assert (frame.unaries == np.log(3)).all()

    # Active probability: 0.333 + (1 - 0.333) * 0.5 = 0.666667
    f = np.array([0, 1, 2], np.int32)
    frame.set_mask(f, 0.5)
    unaries = frame.unaries
    exp_unaries = -np.log(
        np.array(
            [[2/3., 1/6., 1/6.],
             [1/6., 2/3., 1/6.],
             [1/6., 1/6., 2/3.]]
        )
    )
    assert np.isclose(unaries, exp_unaries).all()


    # Unary: -log(probability)
    prob = np.array(
        [[0.7, 0.5, 0.1],
         [0.1, 0.3, 0.05],
         [0.2, 0.2, 0.75]],
        np.float32
    )
    frame.set_proba(prob)
    assert np.isclose(frame.unaries, -np.log(prob)).all()


def test_initial_inferred():
    crf = SimpleCRF(3, 3)
    frame = crf.push_frame()

    frame.set_unbiased()
    assert (frame.get_inferred() == 0).all()
    frame.reset_inferred()
    assert np.isclose(frame.get_inferred(), 1 / 3.).all()

