import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
spec = importlib.util.spec_from_file_location(
    "fetch", Path(__file__).resolve().parents[1] / "scripts" / "fetch_human_corpus.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def test_truncate_stops_at_sentence_end():
    text = "Un deux trois. Quatre cinq six. Sept huit neuf."
    assert fetch.truncate(text, 7) == "Un deux trois. Quatre cinq six."
    assert fetch.truncate(text, 1) == "Un deux trois."  # never empty


def test_balanced_sample_round_robins_domains():
    df = pd.DataFrame({"domain": ["news"] * 50 + ["reddit"] * 5, "text": ["x"] * 55})
    s = fetch.balanced_sample(df, 10, seed=0)
    assert len(s) == 10 and (s["domain"] == "reddit").sum() == 5
