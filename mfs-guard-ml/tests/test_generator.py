import pandas as pd
from src.data_generator import generate
from src.data_validation import validate, MANDATORY


def test_generator_schema_and_invariants(dataset):
    assert set(MANDATORY) <= set(dataset)
    assert validate(dataset, len(dataset))['status'] == 'PASSED'
    assert dataset.user_id.nunique() < len(dataset)/20
    assert dataset.fraud_type.nunique() == 10


def test_reproducible():
    pd.testing.assert_frame_equal(generate(300, 123, False), generate(300, 123, False))
