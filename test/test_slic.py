import pytest
import os
import numpy as np
import ctypes as C

from fast_slic import Slic
from fast_slic.avx2 import SlicAvx2
from PIL import Image

data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
)

@pytest.fixture
def fish_image():
    with Image.open(os.path.join(data_dir, "fish.jpg")) as img:
        return np.array(img)

@pytest.fixture
def fish_image_result():
    with Image.open(os.path.join(data_dir, "fish_result.png")) as img:
        return np.array(img)

@pytest.fixture
def fish_image_avx2_result():
    with Image.open(os.path.join(data_dir, "fish_result.avx2.png")) as img:
        return np.array(img)

@pytest.fixture
def fish_image_01_result():
    with Image.open(os.path.join(data_dir, "fish_result.min_size_factor-0.1.png")) as img:
        return np.array(img)

@pytest.fixture
def fish_image_01_avx2_result():
    with Image.open(os.path.join(data_dir, "fish_result.min_size_factor-0.1.avx2.png")) as img:
        return np.array(img)


def test_slic_model_clusters():
    x = np.zeros([480, 640, 3], np.uint8)
    slic = Slic(num_components=100)
    slic.iterate(x)
    for i, cluster in enumerate(slic.slic_model.clusters):
        assert cluster['number'] == i
        assert isinstance(cluster, dict)
        assert len(cluster['yx']) == 2
        assert isinstance(cluster['yx'], tuple)
        assert len(cluster['color']) == 3
        assert isinstance(cluster['color'], tuple)
        assert isinstance(cluster['num_members'], int)



def test_slic_model_clusters_setter():
    x = np.zeros([480, 640, 3], np.uint8)
    slic = Slic(num_components=100)
    slic.iterate(x)
    orig_clusters = slic.slic_model.clusters
    slic.slic_model.clusters = orig_clusters[:10]
    assert len(slic.slic_model.clusters) == 10
    assert slic.slic_model.clusters == orig_clusters[:10]
    assert slic.slic_model.num_components == 10
    assert slic.num_components == 10


def test_slic(fish_image, fish_image_result, fish_image_avx2_result, fish_image_01_result, fish_image_01_avx2_result):
    assert (Slic(num_components=256, min_size_factor=0).iterate(fish_image) == fish_image_result).all()
    assert (SlicAvx2(num_components=256, min_size_factor=0).iterate(fish_image) == fish_image_avx2_result).all()
    assert (Slic(num_components=256, min_size_factor=0.1).iterate(fish_image) == fish_image_01_result).all()
    assert (SlicAvx2(num_components=256, min_size_factor=0.1).iterate(fish_image) == fish_image_01_avx2_result).all()


