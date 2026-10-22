import json
from pathlib import Path

import pytest

from detbench.briefs import load_briefs
from detbench.detectors.base import DetectionResult, Detector, SentenceScore
from detbench.detectors.stylometry import style_metrics
from detbench.report import build_report
from detbench.revise import rank_sentences, revise
from detbench.styles import load_styles
from detbench.text_utils import split_sentences

ROOT = Path(__file__).resolve().parents[1]
TEXT = ("Le service ouvre lundi. Il compte 300 vélos, dont 120 électriques ! "
        "« C'est enfin là », dit Marie.\nDeuxième paragraphe ici. Et une fin?")


def test_split_sentences_spans_match_text():
    spans = split_sentences(TEXT)
    assert [s for s, _, _ in spans][0] == "Le service ouvre lundi."
    assert len(spans) == 5
    for s, a, b in spans:
        assert TEXT[a:b] == s


def test_style_metrics_counts_ai_phrases():
    m = style_metrics("En conclusion, il est important de noter que tout va bien. Oui.")
    assert m["n_sentences"] == 2
    assert m["ai_phrase_hits"]["en conclusion"] == 1
    assert m["burstiness"] > 0


def test_styles_and_briefs_load():
    styles = load_styles(ROOT / "styles")
    assert {"chroniqueur", "technicien", "conteur"} <= set(styles)
    prompt = styles["technicien"].system_prompt()
    assert "technicien" in prompt and "de plus" in prompt
    briefs = load_briefs(ROOT / "data" / "briefs")
    assert any("300 vélos" in f for b in briefs for f in b.facts)


class FakeDetector(Detector):
    """Flags any sentence containing 'plat' until none remain."""
    name = "fake"
    supports_sentences = True

    def detect(self, text):
        sents = [SentenceScore(s, a, b, 1.0 if "plat" in s else 0.0) for s, a, b in split_sentences(text)]
        flagged = any(s.ai_score for s in sents)
        return DetectionResult(self.name, 0.9 if flagged else 0.1, flagged, {}, sents)


class FakeLLM:
    model = "fake"

    def complete_json(self, prompt, system=""):
        n = prompt.count('"') // 2
        return {str(i): "Phrase réécrite." for i in range(1, n + 1)}


def test_rank_sentences_puts_flagged_first():
    text = "Une phrase vive. Une phrase plate. Encore une vive."
    ranked = rank_sentences(text, [FakeDetector().detect(text)])
    assert ranked[0][1] == "Une phrase plate."


def test_revise_rewrites_only_flagged_sentences_and_stops():
    style = load_styles(ROOT / "styles")["technicien"]
    text = "Une phrase vive. Une phrase plate. Encore une vive. Et une autre vive. Fin vive."
    out, trace = revise(text, FakeLLM(), style, [FakeDetector()], max_rounds=3, fraction=0.2)
    assert "plate" not in out
    assert out.startswith("Une phrase vive.") and out.endswith("Fin vive.")
    assert trace.rounds[-1]["flagged"] == []


def test_revise_quality_gate_reverts():
    style = load_styles(ROOT / "styles")["technicien"]
    text = "Une phrase vive. Une phrase plate. Encore une vive."
    out, trace = revise(text, FakeLLM(), style, [FakeDetector()], max_rounds=2,
                        similarity_fn=lambda a, b: 0.0)
    assert out == text
    assert all(r.get("reverted") for r in trace.rounds[:-1])


def test_report(tmp_path):
    rec = lambda cond, p, flag: {"condition": cond, "style": None, "brief": "b", "text": "x",
                                 "detectors": {"fake": {"p_ai": p, "flagged": flag}},
                                 "quality": {"fact_recall": 1.0, "clarte": 4, "naturel": 4}}
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in
                                                   [rec("raw", 0.9, True), rec("raw", 0.8, True)]))
    (tmp_path / "human.jsonl").write_text(json.dumps(
        {"kind": "human", "condition": "humain", "style": "corpus",
         "detectors": {"fake": {"p_ai": 0.1, "flagged": False}}}))
    md = build_report(tmp_path).read_text()
    assert "| raw | — | 2 | 100 % |" in md
    assert "| raw | 1.00 |" in md


def test_pangram_parsing(tmp_path, monkeypatch):
    from detbench.detectors import pangram

    calls = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def post(url, headers, json, timeout):
        calls.append(url)
        return Resp({"task_id": "t1"})

    def get(url, headers, timeout):
        return Resp({"stage": "STAGE_SUCCESS", "version": "4", "prediction_short": "Human",
                     "fraction_ai": 0.0, "fraction_ai_assisted": 0.2, "fraction_human": 0.8,
                     "windows": [{"text": "a", "start_index": 0, "end_index": 1, "ai_assistance_score": 0.1}]})

    monkeypatch.setattr(pangram.requests, "post", post)
    monkeypatch.setattr(pangram.requests, "get", get)
    det = pangram.PangramDetector(api_key="k", cache_dir=str(tmp_path), daily_word_budget=5)
    r = det.detect("deux mots")
    assert r.ai_probability == pytest.approx(0.1) and not r.flagged
    det.detect("deux mots")  # cached: no second call
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="Budget"):
        det.detect("un texte qui dépasse le budget")


def test_revise_keeps_going_while_a_document_level_detector_flags():
    class DocOnly(Detector):
        name = "doc"
        calls = 0

        def detect(self, text):
            DocOnly.calls += 1
            return DetectionResult(self.name, 0.9, DocOnly.calls < 3, {})

    style = load_styles(ROOT / "styles")["technicien"]
    text = "Une phrase vive. Une phrase plate. Encore une vive."
    _, trace = revise(text, FakeLLM(), style, [FakeDetector(), DocOnly()], max_rounds=5)
    assert trace.rounds[-1]["flagged"] == [] and len(trace.rounds) == 3
