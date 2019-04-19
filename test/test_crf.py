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
    
    import gc; gc.collect()
    # frame should be alive
    frame.unaries
    frame.get_inferred()

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
         [0.1, 0.3, 0.15],
         [0.2, 0.2, 0.75]],
        np.float32
    )
    frame.set_proba(prob)
    assert np.isclose(frame.unaries, -np.log(prob)).all()


def test_proba():
    crf = SimpleCRF(3, 3)
    frame = crf.push_frame()
    # Unary: -log(probability)
    prob = np.array(
        [[0.7, 0.5, 0.1],
         [0.1, 0.3, 0.15],
         [0.2, 0.2, 0.75]],
        np.float32
    )
    frame.set_proba(prob)

    assert np.isclose(frame.get_inferred(), 0).all()
    crf.initialize()
    assert np.isclose(frame.get_inferred(), prob).all()


def test_initial_inferred():
    crf = SimpleCRF(3, 3)
    frame = crf.push_frame()

    frame.set_unbiased()
    assert (frame.get_inferred() == 0).all()
    frame.reset_inferred()
    assert np.isclose(frame.get_inferred(), 1 / 3.).all()


def test_set_yxmrgb():
    crf = SimpleCRF(3, 3)
    frame = crf.push_frame()
    frame.set_yxmrgb(
        np.array(
            [
                [1,2,1,3,4,5],
                [6,7,2,8,9,10],
                [11,12,3, 13,14,15],
            ],
            np.int32,
        )
    )
    res = frame.get_yxmrgb()
    assert len(res) == 3
    assert res[0] == [1,2,1,3,4,5]
    assert res[1] == [6,7,2,8,9,10]
    assert res[2] == [11,12,3, 13,14,15]


def test_set_connectivity():
    crf = SimpleCRF(3, 3)
    frame = crf.push_frame()
    assert frame.get_connectivity() == [[], [], []]
    with pytest.raises(TypeError):
        frame.set_connectivity([None, None, None])
    frame.set_connectivity([[0, 1], [2], [0]])
    assert frame.get_connectivity() == [[0, 1], [2], [0]]


def test_spatial_energy():
    # weight * exp(-(rgb_i - rgb_j) ** 2 / (2 * srgb ** 2) + -(xy_i - xy_j) ** 2 / (2 * sxy ** 2))
    spatial_srgb = 3.5
    spatial_w = 1.9
    spatial_sxy = 2.4
    crf = SimpleCRF(3, 2)

    crf.spatial_srgb = spatial_srgb
    crf.spatial_w = spatial_w
    crf.spatial_sxy = spatial_sxy
    assert np.isclose(crf.spatial_srgb, spatial_srgb)
    assert np.isclose(crf.spatial_w, spatial_w)
    assert np.isclose(crf.spatial_sxy, spatial_sxy)

    frame = crf.push_frame()

    frame.set_yxmrgb(
        np.array(
            [
                [1, 1, 1, 1, 2, 6],
                [0, 0, 1, 4, 5, 3],
            ],
            np.int32,
        )
    )

    energy = spatial_w * np.exp(-((1 - 4) ** 2 + (2 - 5) ** 2 + (6 - 3) ** 2) / (2 * spatial_srgb ** 2) - ((1 - 0) ** 2 + (1 - 0)**2) / (2 * spatial_sxy ** 2))

    assert np.isclose(frame.spatial_pairwise_energy(0, 1), energy)
    assert np.isclose(frame.spatial_pairwise_energy(1, 0), energy)
    assert np.isclose(frame.spatial_pairwise_energy(0, 0), 0)
    assert np.isclose(frame.spatial_pairwise_energy(1, 1), 0)

    
def test_temporal_energy():
    # weight * exp(-(rgb_i(t) - rgb_i(T)) ** 2 / (2 * srgb ** 2)) 

    temporal_srgb = 3.5
    temporal_w = 1.9
    crf = SimpleCRF(3, 1)

    crf.temporal_srgb = temporal_srgb
    crf.temporal_w = temporal_w

    assert np.isclose(crf.temporal_srgb, temporal_srgb)
    assert np.isclose(crf.temporal_w , temporal_w)

    frame_1 = crf.push_frame()
    frame_2 = crf.push_frame()

    frame_1.set_yxmrgb(
        np.array(
            [
                [0, 0, 1, 1, 2, 6],
            ],
            np.int32,
        )
    )
    frame_2.set_yxmrgb(
        np.array(
            [
                [0, 0, 1, 4, 5, 3],
            ],
            np.int32,
        )
    )

    energy = temporal_w * np.exp(-(((1 - 4) ** 2 + (2 - 5) ** 2 + (6 - 3) ** 2) / (2 * temporal_srgb ** 2)))

    assert np.isclose(frame_1.temporal_pairwise_energy(0, frame_2), energy)
    assert np.isclose(frame_2.temporal_pairwise_energy(0, frame_1), energy)
    assert np.isclose(frame_1.temporal_pairwise_energy(0, frame_1), 0)
    assert np.isclose(frame_2.temporal_pairwise_energy(0, frame_2), 0)

