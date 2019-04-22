import pytest
import numpy as np

from fast_slic import Slic

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

