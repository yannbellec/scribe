"""Plumbing test of Binoculars / Fast-DetectGPT with tiny random models (no download)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_pair(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("tiny")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["Le service ouvre lundi avec trois cents vélos.", "Une phrase plate. Une autre phrase."] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<eos>"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>")
    paths = []
    for i, name in enumerate(("obs", "perf")):
        torch.manual_seed(i)
        cfg = GPT2Config(vocab_size=len(fast), n_positions=128, n_embd=32, n_layer=1, n_head=2)
        d = root / name
        GPT2LMHeadModel(cfg).save_pretrained(d)
        fast.save_pretrained(d)
        paths.append(str(d))
    return paths


def test_detectors_produce_scores_and_sentence_spans(tiny_pair):
    from detbench.detectors.binoculars import BinocularsDetector
    from detbench.detectors.fast_detectgpt import FastDetectGPTDetector

    obs, perf = tiny_pair
    text = "Le service ouvre lundi. Une phrase plate avec trois cents vélos. Une autre phrase."
    for cls in (BinocularsDetector, FastDetectGPTDetector):
        det = cls(observer=obs, performer=perf, quantization=None, max_tokens=64)
        r = det.detect(text)
        assert 0.0 <= r.ai_probability <= 1.0
        assert r.raw["score"] == r.raw["score"]  # not NaN
        assert r.sentences and all(text[s.start:s.end] == s.text for s in r.sentences)


def test_long_text_is_windowed(tiny_pair):
    from detbench.detectors.binoculars import BinocularsDetector

    obs, perf = tiny_pair
    text = " ".join(["Une phrase plate avec trois cents vélos."] * 40)
    det = BinocularsDetector(observer=obs, performer=perf, quantization=None, max_tokens=32)
    st = det.pair.stats(text)
    assert len(st.offsets) == len(st.ce) > 32
    assert st.offsets[-1][1] <= len(text)


def test_editlens_score_is_expected_bucket():
    from detbench.detectors.editlens import EditLensDetector

    class Out:
        logits = torch.tensor([[0.0, 0.0, 10.0]])  # almost all mass on the last bucket

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, **kw):
            return Out()

    det = EditLensDetector.__new__(EditLensDetector)
    det.tokenizer = lambda text, **kw: {"input_ids": torch.tensor([[1, 2]])}
    det.model, det.max_length, det.threshold = FakeModel(), 16, 0.5
    r = det.detect("texte")
    assert r.ai_probability > 0.99 and r.flagged and r.raw["bucket"] == 2


def test_mirror_verification(tmp_path):
    from detbench.detectors.editlens import ADAPTER_FILE, sha256_file, verify_mirror

    (tmp_path / ADAPTER_FILE).write_bytes(b"weights")
    good = sha256_file(tmp_path / ADAPTER_FILE)
    verify_mirror("pangram/x", str(tmp_path), expected=good)
    with pytest.raises(RuntimeError, match="refusé"):
        verify_mirror("pangram/x", str(tmp_path), expected="0" * 64)


def test_editlens_reads_normed_head(tmp_path):
    from safetensors.torch import save_file

    from detbench.detectors.editlens import EditLensDetector, NormedLinear, clean_text

    path = tmp_path / "adapter_model.safetensors"
    save_file({"base_model.model.score.modules_to_save.default.linear.weight": torch.zeros(7, 8),
               "base_model.model.score.modules_to_save.default.norm.weight": torch.ones(8)}, str(path))
    assert EditLensDetector._num_buckets(path) == 7
    assert NormedLinear(8, 7)(torch.zeros(2, 8)).shape == (2, 7)
    assert clean_text("<think>x</think> Hello\n\n World ") == "hello world"
