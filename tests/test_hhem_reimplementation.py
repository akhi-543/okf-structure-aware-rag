"""HHEM-2.1-Open loaded without its remote code must reproduce the model card's scores.

Integration test (downloads the gated checkpoint; needs HF_TOKEN): pytest -m integration
"""
import pytest

PAIRS = [
    ("The capital of France is Berlin.", "The capital of France is Paris."),
    ("I am in California", "I am in United States."),
    ("I am in United States", "I am in California."),
    ("A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."),
    ("A boy is jumping on skateboard in the middle of a red bridge.", "The boy skates down the sidewalk on a red bridge"),
    ("A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
     "A blond man wearing a brown shirt is reading a book."),
    ("Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."),
]
MODEL_CARD = [0.0111, 0.6474, 0.1290, 0.8969, 0.1846, 0.0050, 0.0543]


@pytest.mark.integration
def test_hhem_matches_model_card():
    from scripts.llm_brightmart_v4 import hhem_predict, load_hhem
    tok, model = load_hhem()
    assert [round(x, 4) for x in hhem_predict(tok, model, PAIRS)] == MODEL_CARD
