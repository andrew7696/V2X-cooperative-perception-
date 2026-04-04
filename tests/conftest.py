import pytest

def pytest_collection_modifyitems(config, items):
    """Skip all tests if torch is not installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        skip_torch = pytest.mark.skip(reason="torch not installed")
        for item in items:
            item.add_marker(skip_torch)
