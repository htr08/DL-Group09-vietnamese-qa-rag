from src.metrics import recall_at_k, mrr_at_k, mean_metric


def test_recall_at_k():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    assert recall_at_k(retrieved, "doc_1", k=1) == 1.0
    assert recall_at_k(retrieved, "doc_3", k=3) == 1.0
    assert recall_at_k(retrieved, "doc_4", k=3) == 0.0
    assert recall_at_k(retrieved, "doc_non_exist", k=5) == 0.0


def test_mrr_at_k():
    retrieved = ["doc_1", "doc_2", "doc_3"]
    assert mrr_at_k(retrieved, "doc_1", k=10) == 1.0
    assert mrr_at_k(retrieved, "doc_2", k=10) == 0.5
    assert mrr_at_k(retrieved, "doc_3", k=10) == 1.0 / 3.0
    assert mrr_at_k(retrieved, "doc_4", k=10) == 0.0


def test_mean_metric():
    assert mean_metric([1.0, 0.5, 0.0]) == 0.5
    assert mean_metric([]) == 0.0