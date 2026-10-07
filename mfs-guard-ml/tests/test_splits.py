import numpy as np
from src.split_data import split


def test_splits_are_disjoint_chronological_and_complete(dataset):
    indexes, _ = split(dataset)
    joined = np.concatenate(list(indexes.values()))
    assert np.array_equal(joined, np.arange(len(dataset)))
    assert len(indexes['train']) == int(.7*len(dataset))
    for a, b in zip(list(indexes.values())[:-1], list(indexes.values())[1:]):
        assert dataset.iloc[a[-1]].timestamp < dataset.iloc[b[0]].timestamp
