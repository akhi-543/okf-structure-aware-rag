import numpy as np
from okf_web_bench.okf_web_bench_models import (
    BGE_QUERY_PREFIX,
    health_payload,
    device_label,
    prefixed_query,
    BgeQueryEncoder,
)

def test_device_label_cpu_when_cuda_false():
    assert device_label(False, 0) == "cpu"

def test_device_label_cuda_n():
    assert device_label(True, 0) == "cuda:0"
    assert device_label(True, 1) == "cuda:1"

def test_health_payload_independent_flags():
    h = health_payload(postgres=True, bge=False, qwen=True, cuda=False, cuda_index=None)
    assert h == {
        "postgres": True,
        "bge": False,
        "qwen": True,
        "cuda": False,
        "device": "cpu",
    }

def test_prefixed_query_trailing_space():
    assert prefixed_query("How?") == BGE_QUERY_PREFIX + "How?"
    assert BGE_QUERY_PREFIX.endswith(" ")

def test_bge_encoder_embed_fn_list_in():
    def embed_fn(texts):
        assert texts == [BGE_QUERY_PREFIX + "q"]
        return np.ones((1, 768), dtype=np.float32)

    enc = BgeQueryEncoder(embed_fn=embed_fn, available=True)
    out = enc.encode(["q"])
    assert out.shape == (1, 768)


def test_encoder_does_not_double_prefix():
    seen = []

    def embed_fn(texts):
        seen.extend(texts)
        return np.ones((len(texts), 768), np.float32)

    BgeQueryEncoder(embed_fn=embed_fn, available=True).encode(["q"])
    assert seen == ["Represent this sentence for searching relevant passages: q"]


def test_qwen_load_dtype_cpu_is_float32():
    import torch
    from okf_web_bench.okf_web_bench_models import qwen_load_dtype

    assert qwen_load_dtype(False) == torch.float32


def test_bge_encode_returns_float32_even_if_embed_fn_does_not():
    def embed_fn(texts):
        return np.ones((1, 768), dtype=np.float64)

    out = BgeQueryEncoder(embed_fn=embed_fn, available=True).encode(["q"])
    assert out.dtype == np.float32


def test_qwen_generate_uses_non_thinking(monkeypatch):
    from okf_web_bench.okf_web_bench_models import QwenGenerator, SYSTEM_PROMPT

    class Tok:
        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        ):
            assert enable_thinking is False
            assert messages[0]["content"] == SYSTEM_PROMPT
            return "PROMPT"

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3]

        def decode(self, ids, skip_special_tokens=True):
            return "decoded-answer"

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            assert kwargs.get("do_sample") is False
            assert kwargs.get("max_new_tokens") == 256

            class T:
                def __getitem__(self, i):
                    return [0, 0, 0, 9, 9]

            return T()

    g = QwenGenerator(tokenizer=Tok(), model=Model(), available=True)
    assert g.generate("q", "[1] p\nhi") == "decoded-answer"


def test_store_sql_is_select_only():
    import inspect

    import okf_web_bench.okf_web_bench_store as store_mod

    src = inspect.getsource(store_mod)
    assert "UPDATE" not in src
    assert "INSERT" not in src
    assert "DELETE" not in src
    for name, val in vars(store_mod).items():
        if name.endswith("_SQL") and isinstance(val, str):
            first = val.strip().split()[0].upper()
            assert first == "SELECT", f"{name} must be SELECT-only, got {first}"


def test_store_available_arms_without_bm25(monkeypatch):
    from okf_web_bench.okf_web_bench_models import BgeQueryEncoder
    from okf_web_bench.okf_web_bench_store import WebBenchStore
    import okf_web_bench.okf_web_bench_store as store_mod

    monkeypatch.setattr(store_mod, "_import_bm25_index", lambda: None)
    store = WebBenchStore(lambda: None, BgeQueryEncoder(available=True))
    assert store.available_arms() == ["flat"]  # structured lane needs BM25


def test_store_available_arms_with_bm25():
    from okf_web_bench.okf_web_bench_models import BgeQueryEncoder
    from okf_web_bench.okf_web_bench_store import WebBenchStore

    store = WebBenchStore(lambda: None, BgeQueryEncoder(available=True))
    assert store.available_arms() == ["flat", "structured"]


def test_embedding_to_array_accepts_pgvector_vector():
    import numpy as np
    from okf_web_bench.okf_web_bench_store import embedding_to_array

    class Vector:
        def __init__(self, data):
            self._data = data

        def to_list(self):
            return list(self._data)

        def __float__(self):
            raise TypeError(
                "float() argument must be a string or a real number, not 'Vector'"
            )

    vec = Vector([0.0] * 767 + [1.0])
    arr = embedding_to_array(vec)
    assert arr.shape == (768,)
    assert arr.dtype == np.float32
    assert float(arr[-1]) == 1.0
    stacked = np.stack([embedding_to_array(vec), embedding_to_array(vec)])
    assert stacked.shape == (2, 768)
