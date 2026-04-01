import pytest
import numpy as np


#def test_math_consistency(backend):
#    # Dati di test
#    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
#    tensor = backend.numpy_to_tensor(data)
#    
#    # Test della somma
#    res = backend.sum(tensor)
#    assert np.isclose(backend.tensor_to_numpy(res), 6.0)
#
#def test_slicing_and_scatter(backend):
#    # Creiamo un volume 3D di zeri (10, 10, 10)
#    canvas = backend.zeros((10, 10, 10))
#    # Creiamo una patch di uni (4, 4, 4)
#    patch = backend.numpy_to_tensor(np.ones((4, 4, 4), dtype=np.float32))
#    
#    # Test scatter_update
#    starts = (2, 2, 2)
#    canvas = backend.scatter_update(canvas, patch, starts)
#    
#    # Test slice (estraiamo esattamente dove abbiamo scritto)
#    extracted = backend.slice(canvas, starts, (4, 4, 4))
#    
#    # Verifica
#    extracted_np = backend.tensor_to_numpy(extracted)
#    assert np.all(extracted_np == 1.0)
#    
#    # Verifica che fuori dalla patch ci siano ancora zeri
#    edge_case = backend.slice(canvas, (0,0,0), (1,1,1))
#    assert backend.tensor_to_numpy(edge_case) == 0.0



class BaseBackendTest:
    """Classe base che contiene test comuni a tutti i backend."""
    
    def test_sum_consistency(self, backend):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = backend.numpy_to_tensor(data)
        res = backend.sum(tensor)
        assert np.isclose(backend.tensor_to_numpy(res), 6.0)

    def test_slice_shape(self, backend):
        shape = (10, 10, 10)
        data = backend.zeros(shape, dtype="float32")
        patch = backend.slice(data, (2, 2, 2), (4, 4, 4))
        assert tuple(patch.shape) == (4, 4, 4)