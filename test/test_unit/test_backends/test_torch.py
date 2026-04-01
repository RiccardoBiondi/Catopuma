import torch
import numpy as np
import pytest
from .test_common import BaseBackendTest
from catopuma.backends.factory import get_backend, clear_backend





class TestTorchBackend(BaseBackendTest):
    """Eredita i test comuni e aggiunge test specifici per PyTorch."""

    @pytest.fixture
    def backend(self):
        """Inietta il backend specifico per questa classe di test."""
        clear_backend()
        return get_backend("torch")

    # --- Test Specifici solo per Torch ---

    def test_autograd_support(self, backend):
        """Verifica che le operazioni del backend mantengano i gradienti."""
        data = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        # Usiamo il metodo sum ereditato concettualmente ma via backend
        res = backend.sum(backend.pow(data, 2.0))
        res.backward()
        
        assert data.grad is not None
        assert torch.allclose(data.grad, torch.tensor([2.0, 4.0, 6.0]))

    def test_cuda_transfer(self, backend):
        """Verifica lo spostamento su GPU se disponibile."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA non disponibile")
            
        data = torch.randn((5, 5)).cuda()
        # Il backend deve saper gestire un tensore che è già su GPU
        res = backend.abs(data)
        assert res.is_cuda

    def test_torch_gradient_flow(self, backend):

        # Creiamo un tensore con gradienti abilitati
        data = torch.tensor([2.0, 4.0], requires_grad=True)

        # Eseguiamo operazioni tramite il backend
        # Esempio: f(x) = sum(x^2) -> f'(x) = 2x
        x_pow = backend.pow(data, 2.0)
        loss = backend.sum(x_pow)

        # Backpropagation
        loss.backward()

        # Verifica dei gradienti: grad(2^2) = 4, grad(4^2) = 8
        expected_grad = torch.tensor([4.0, 8.0])
        assert torch.allclose(data.grad, expected_grad)



    def test_torch_gpu_tensor_to_numpy(self, backend):

        data = torch.tensor([2.0, 4.0], requires_grad=True)
        data.to("cuda")

        np_version = backend.tensor_to_numpy(data)

        manual_np = data.detach().cpu().numpy()
        assert all(np.isclose(manual_np, np_version))
