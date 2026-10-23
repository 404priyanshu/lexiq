import pytest

from app.ingest.chunker import chunk_pages, join_pages


def words(texts: list[str]) -> list[int]:
    """Deterministic stand-in for the tokenizer: one token per word."""
    return [len(t.split()) for t in texts]


def sentence(i: int, n: int = 10) -> str:
    return f"Sentence {i} " + " ".join(["word"] * (n - 3)) + " end."


def test_empty_document_has_no_chunks():
    assert chunk_pages([""], words) == []


def test_spans_slice_back_to_content_and_pages_are_correct():
    pages = [
        "Page one intro.\n\n" + " ".join(sentence(i) for i in range(30)),
        " ".join(sentence(i) for i in range(30, 60)),
        "Short last page.",
    ]
    text, _ = join_pages(pages)
    chunks = chunk_pages(pages, words, target=100, max_tokens=120, overlap=20)

    assert len(chunks) > 3
    for c in chunks:
        assert text[c.char_start : c.char_end] == c.content
        assert 1 <= c.page_start <= c.page_end <= 3
        assert c.token_count <= 120
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 3
    assert "Short last page." in chunks[-1].content
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_every_sentence_is_covered_and_neighbours_overlap():
    pages = [" ".join(sentence(i) for i in range(50))]
    chunks = chunk_pages(pages, words, target=60, max_tokens=80, overlap=20)

    for i in range(50):
        assert any(f"Sentence {i} " in c.content for c in chunks), i
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.char_start < a.char_end, "expected overlap between consecutive chunks"
        assert b.char_end > a.char_end, "each chunk must make progress"


def test_unbroken_text_is_split_by_words_under_max():
    pages = [" ".join(f"w{i}" for i in range(1000))]  # one 1000-word "sentence"
    chunks = chunk_pages(pages, words, target=100, max_tokens=120, overlap=0)

    assert all(c.token_count <= 120 for c in chunks)
    assert sum(len(c.content.split()) for c in chunks) == 1000


def test_rejects_inconsistent_limits():
    with pytest.raises(ValueError):
        chunk_pages(["x"], words, target=100, max_tokens=50)


def test_real_tokenizer_keeps_chunks_under_model_limit(models):
    pages = [" ".join(sentence(i, 25) for i in range(400))]
    chunks = chunk_pages(pages, models.count_tokens, target=400, max_tokens=480, overlap=64)

    actual = models.count_tokens([c.content for c in chunks])
    # +2 for [CLS]/[SEP]; must fit the 512-token embedder window.
    assert max(actual) + 2 <= 512
