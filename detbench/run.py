"""Benchmark CLI.

    python -m detbench.run human        # score the human corpus (false positives)
    python -m detbench.run calibrate    # set local thresholds at the target FPR
    python -m detbench.run generate     # briefs × styles × conditions → results/runs.jsonl
    python -m detbench.run report       # results/report.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from .briefs import load_briefs
from .detectors import build_detectors
from .detectors.stylometry import style_metrics
from .llm import Claude
from .revise import revise
from .styles import load_styles

CONDITIONS = ("raw", "style", "style_revise")


def load_env(path: str = ".env") -> None:
    import os
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    calib = Path(cfg.get("results_dir", "results")) / "calibration.json"
    if calib.exists():  # thresholds measured on the human corpus override defaults
        for name, thr in json.loads(calib.read_text()).items():
            if name in (cfg.get("detectors") or {}):
                cfg["detectors"][name]["threshold"] = thr
    return cfg


def _split(detectors, cfg):
    external = set(cfg.get("external_detectors", ["pangram"]))
    local = [d for d in detectors if d.name not in external]
    ext = [d for d in detectors if d.name in external]
    return local, ext


def _score(detectors, text) -> dict:
    out = {}
    for d in detectors:
        try:
            r = d.detect(text)
            out[r.detector] = {"p_ai": round(r.ai_probability, 4), "flagged": r.flagged, **r.raw}
        except Exception as e:  # one failing detector must not kill the run
            out[d.name] = {"error": str(e)}
    return out


def cmd_human(cfg, args):
    detectors = build_detectors(cfg)
    local, ext = _split(detectors, cfg)
    use = local + (ext if args.external else [])
    out = Path(cfg["results_dir"]) / "human.jsonl"
    with out.open("a", encoding="utf-8") as f:
        for p in sorted(Path(cfg["human_dir"]).glob("*.txt")):
            text = p.read_text(encoding="utf-8").strip()
            rec = {"source": p.name, "kind": "human", "text": text, "metrics": style_metrics(text),
                   "detectors": _score(use, text)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(p.name, {k: v.get("p_ai") for k, v in rec["detectors"].items()})


def cmd_calibrate(cfg, args):
    """Threshold = (1 - target_fpr) quantile of human scores, per local detector."""
    import numpy as np
    recs = [json.loads(l) for l in (Path(cfg["results_dir"]) / "human.jsonl").open(encoding="utf-8")]
    calib = {}
    for name, sign in (("binoculars", -1), ("fast_detectgpt", 1)):
        scores = [r["detectors"][name]["score"] for r in recs if "score" in r["detectors"].get(name, {})]
        if len(scores) < 10:
            print(f"{name} : seulement {len(scores)} textes humains, il en faut au moins 10")
            continue
        q = np.quantile(np.array(scores) * sign, 1 - cfg.get("target_fpr", 0.01))
        calib[name] = float(q * sign)
    path = Path(cfg["results_dir"]) / "calibration.json"
    path.write_text(json.dumps(calib, indent=2))
    print("Seuils calibrés :", calib)


def cmd_generate(cfg, args):
    llm = Claude(**cfg.get("claude", {}))
    styles = load_styles(cfg["styles_dir"])
    briefs = load_briefs(cfg["briefs_dir"])
    if args.style:
        styles = {k: v for k, v in styles.items() if k in args.style}
    if args.brief:
        briefs = [b for b in briefs if b.id in args.brief]
    conditions = args.condition or list(CONDITIONS)
    detectors = build_detectors(cfg)
    local, ext = _split(detectors, cfg)
    sim_fn = None
    if cfg.get("revision", {}).get("min_similarity"):
        from .quality import semantic_similarity
        sim_fn = semantic_similarity

    out = Path(cfg["results_dir"]) / "runs.jsonl"
    for brief in briefs:
        for cond in conditions:
            style_items = [(None, None)] if cond == "raw" else list(styles.items())
            for style_name, style in style_items:
                for seed in range(args.repeats):
                    system = style.system_prompt() if style else ""
                    draft = llm.complete(brief.prompt(), system)
                    text, trace = draft, None
                    if cond == "style_revise":
                        text, trace = revise(draft, llm, style, local,
                                             similarity_fn=sim_fn, **cfg.get("revision", {}))
                    rec = {
                        "time": dt.datetime.now().isoformat(timespec="seconds"),
                        "brief": brief.id, "condition": cond, "style": style_name, "repeat": seed,
                        "model": llm.model, "text": text,
                        "draft": draft if cond == "style_revise" else None,
                        "revision": asdict(trace) if trace else None,
                        "metrics": style_metrics(text),
                        "detectors": _score(local + (ext if not args.no_external else []), text),
                    }
                    if not args.no_judge:
                        from .quality import judge
                        rec["quality"] = judge(llm, brief, text)
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(brief.id, cond, style_name, seed,
                          {k: v.get("p_ai") for k, v in rec["detectors"].items()})


def cmd_report(cfg, args):
    from .report import build_report
    path = build_report(Path(cfg["results_dir"]))
    print("Rapport :", path)


def main():
    ap = argparse.ArgumentParser(prog="detbench")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("human")
    h.add_argument("--external", action="store_true", help="inclure Pangram (consomme le quota)")
    sub.add_parser("calibrate")
    g = sub.add_parser("generate")
    g.add_argument("--brief", nargs="*")
    g.add_argument("--style", nargs="*")
    g.add_argument("--condition", nargs="*", choices=CONDITIONS)
    g.add_argument("--repeats", type=int, default=1)
    g.add_argument("--no-external", action="store_true", help="ne pas appeler Pangram")
    g.add_argument("--no-judge", action="store_true", help="ne pas noter la qualité")
    sub.add_parser("report")
    args = ap.parse_args()
    load_env()
    cfg = load_config(args.config)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    {"human": cmd_human, "calibrate": cmd_calibrate, "generate": cmd_generate, "report": cmd_report}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
