import numpy as np


def test_embeddings_are_normalized_and_rank_relevant_passage_first(models):
    passages = [
        "The indemnification obligations survive termination of this Agreement.",
        "This Agreement is governed by the laws of the State of Delaware.",
        "The cafeteria serves soup on Fridays.",
    ]
    docs = models.embedder.embed_documents(passages)
    query = models.embedder.embed_query("Which state's law governs the contract?")

    assert docs.shape == (3, 384)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)
    assert int(np.argmax(docs @ query)) == 1


def test_batching_does_not_change_results(models):
    texts = [f"clause {i} " * (i + 1) for i in range(10)]
    batched = models.embedder.embed_documents(texts)
    single = np.stack([models.embedder.embed_documents([t])[0] for t in texts])

    assert np.allclose(batched, single, atol=1e-4)
