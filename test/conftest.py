import pytest
from catopuma.backends.factory import get_backend, clear_backend

@pytest.fixture(params=["numpy", "torch"])
def backend(request):
    """Fixture che istanzia a turno ogni backend disponibile."""
    clear_backend() # Reset del Singleton tra un test e l'altro
    try:
        return get_backend(request.param)
    except ImportError:
        pytest.skip(f"Backend {request.param} not installed.")