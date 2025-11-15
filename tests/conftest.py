"""Pytest configuration and fixtures for DoodleHunter tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a sample 128x128 grayscale image."""
    return np.random.rand(128, 128, 1).astype(np.float32)


@pytest.fixture
def sample_batch():
    """Create a batch of sample images."""
    return np.random.rand(32, 128, 128, 1).astype(np.float32)


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def models_dir(project_root):
    """Return the models directory."""
    return project_root / "models"
