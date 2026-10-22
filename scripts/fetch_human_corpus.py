"""Download human-written reference texts (French + English) for the benchmark.

Sources, all written before LLM-generated text became common:
- fr: MLSUM (Scialom et al., EMNLP 2020), ~425k Le Monde articles, 2010-2019.
      Non-commercial research use only. huggingface.co/datasets/reciTAL/mlsum
- en: RAID (Dugan et al., ACL 2024), the human-written split (model == "human",
      no attack) over prose domains: news, reddit, reviews, wiki, abstracts, books. MIT.
      huggingface.co/datasets/liamdugan/raid

The full filtered pool is kept in data/corpora/<lang>.parquet; a balanced random
sample is written as one .txt per text in data/human/<lang>/, with manifest.jsonl.

    python scripts/fetch_human_corpus.py                    # 500 fr + 500 en
    python scripts/fetch_human_corpus.py --lang en --n 2000
    python scripts/fetch_human_corpus.py --fr-split train   # whole MLSUM train pool (~390k)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detbench.text_utils import split_sentences  # noqa: E402

RAID_DOMAINS = ("news", "reddit", "reviews", "wiki", "abstracts", "books")


def truncate(text: str, max_words: int) -> str:
    """Cut at the last sentence end before max_words, so texts match generated lengths."""
    out, n = [], 0
    for sent, _, _ in split_sentences(text):
        w = len(re.findall(r"\w+", sent))
        if n + w > max_words and out:
            break
        out.append(sent)
        n += w
    return " ".join(out)


def query(sql: str):
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con.execute(sql).fetch_df()


def fetch_fr(split: str):
    df = query(f"""
        SELECT url AS id, date, topic AS domain, title, text
        FROM 'hf://datasets/reciTAL/mlsum@~parquet/fr/{split}/*.parquet'
    """)
    df["source"] = "mlsum-fr"
    return df


def fetch_en():
    domains = ", ".join(f"'{d}'" for d in RAID_DOMAINS)
    df = query(f"""
        SELECT id, domain, title, generation AS text
        FROM 'hf://datasets/liamdugan/raid@~parquet/raid/partial-train/*.parquet'
        WHERE model = 'human' AND attack = 'none' AND domain IN ({domains})
    """)
    df = df.drop_duplicates("id")
    df["source"] = "raid"
    return df


def balanced_sample(df, n: int, seed: int):
    """Round-robin over domains so no single genre dominates the sample."""
    rng = random.Random(seed)
    groups = {d: rng.sample(list(g.index), len(g)) for d, g in df.groupby("domain")}
    picked = []
    while len(picked) < n and any(groups.values()):
        for d in list(groups):
            if groups[d] and len(picked) < n:
                picked.append(groups[d].pop())
    return df.loc[picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", nargs="*", default=["fr", "en"], choices=("fr", "en"))
    ap.add_argument("--n", type=int, default=500, help="textes échantillonnés par langue")
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument("--fr-split", default="test", choices=("test", "validation", "train"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for lang in args.lang:
        print(f"[{lang}] téléchargement…", flush=True)
        df = fetch_fr(args.fr_split) if lang == "fr" else fetch_en()
        df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df["text"] = [truncate(t, args.max_words) for t in df["text"]]
        df["n_words"] = df["text"].str.count(r"\w+")
        df = df[df["n_words"] >= args.min_words].reset_index(drop=True)

        pool = ROOT / "data" / "corpora" / f"{lang}.parquet"
        pool.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pool, index=False)
        print(f"[{lang}] {len(df)} textes humains conservés → {pool}", flush=True)

        out = ROOT / "data" / "human" / lang
        out.mkdir(parents=True, exist_ok=True)
        sample = balanced_sample(df, args.n, args.seed)
        with (out / "manifest.jsonl").open("w", encoding="utf-8") as m:
            for i, row in enumerate(sample.itertuples()):
                name = f"{row.source}-{i:05d}.txt"
                (out / name).write_text(row.text + "\n", encoding="utf-8")
                meta = {"file": name, "source": row.source, "id": str(row.id), "domain": row.domain,
                        "title": row.title, "n_words": int(row.n_words)}
                m.write(json.dumps(meta, ensure_ascii=False) + "\n")
        counts = sample["domain"].value_counts().head(8).to_dict()
        print(f"[{lang}] {len(sample)} textes écrits dans {out} — genres : {counts}", flush=True)


if __name__ == "__main__":
    main()
