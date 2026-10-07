import pytest
from src.data_generator import generate


@pytest.fixture(scope='session')
def dataset():
    return generate(6000, seed=42, save=False)
