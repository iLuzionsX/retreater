"""Off-Mac translation eval for Gemma 4 E4B instruction-tuned at 8-bit.

Loads bartowski's Q8_0 GGUF of google/gemma-4-E4B-it (the same checkpoint family
as mlx-community/gemma-4-e4b-it-8bit) and scores the baseline prompt/sampler
against the live prompt/sampler on a public English-Spanish parallel set.

This does not measure Apple MLX or microphone latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts import (
    GEMMA4_CHAT_SAMPLING,
    GREEDY_SAMPLING,
    SamplingConfig,
    build_translate_prompt,
    build_translate_prompt_baseline,
)

TATOEBA_ZIP = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-es.txt.zip"
PAIR_SOURCE = "OPUS Tatoeba v2023-04-12 English-Spanish, sentences of at most 24 words"


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def load_pairs(cache_dir: Path, limit: int) -> list[tuple[str, str]]:
    zip_path = cache_dir / "en-es.txt.zip"
    _download(TATOEBA_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        english_name = next(name for name in names if name.endswith(".en"))
        spanish_name = next(name for name in names if name.endswith(".es"))
        english = archive.read(english_name).decode("utf-8").splitlines()
        spanish = archive.read(spanish_name).decode("utf-8").splitlines()
    if len(english) != len(spanish):
        raise RuntimeError(f"Tatoeba line mismatch: {len(english)} English vs {len(spanish)} Spanish")
    pairs = [
        (source.strip(), target.strip())
        for source, target in zip(english, spanish, strict=True)
        if source.strip() and target.strip() and len(source.split()) <= 24
    ]
    if len(pairs) < limit:
        raise RuntimeError(f"Only {len(pairs)} Tatoeba pairs were usable")
    stride = max(1, len(pairs) // limit)
    return pairs[::stride][:limit]


def _generate(llm: object, prompt: str, sampling: SamplingConfig, max_tokens: int) -> tuple[str, float]:
    started = time.perf_counter()
    result = llm.create_chat_completion(  # type: ignore[attr-defined]
        messages=[{"role": "user", "content": prompt}],
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=max(sampling.top_k, 1),
        max_tokens=max_tokens,
        seed=0,
    )
    elapsed = time.perf_counter() - started
    text = result["choices"][0]["message"]["content"]
    return str(text).strip(), elapsed


def _scores(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"bleu": float(bleu.score), "chrf": float(chrf.score)}


def _summarize(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "mean_seconds": statistics.fmean(times),
        "median_seconds": statistics.median(times),
        "p95_seconds": ordered[p95_index],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to gemma-4-E4B-it Q8_0 GGUF")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/tatoeba-en-es"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/translation-eval.json"))
    args = parser.parse_args()

    from llama_cpp import Llama

    pairs = load_pairs(args.cache_dir, args.limit)
    llm = Llama(
        model_path=str(args.model),
        n_ctx=2048,
        n_threads=args.threads,
        verbose=False,
        chat_format=None,
    )
    conditions = {
        "baseline_prompt_temp1": (build_translate_prompt_baseline, GEMMA4_CHAT_SAMPLING),
        "baseline_prompt_greedy": (build_translate_prompt_baseline, GREEDY_SAMPLING),
        "improved_prompt_temp1": (build_translate_prompt, GEMMA4_CHAT_SAMPLING),
        "improved_prompt_greedy": (build_translate_prompt, GREEDY_SAMPLING),
    }
    report: dict[str, object] = {
        "model": str(args.model),
        "checkpoint": "google/gemma-4-E4B-it Q8_0 via bartowski GGUF",
        "pairs": PAIR_SOURCE,
        "limit": len(pairs),
        "conditions": {},
    }
    for name, (builder, sampling) in conditions.items():
        hypotheses: list[str] = []
        times: list[float] = []
        for source, _reference in pairs:
            prompt = builder(source, "English", "Spanish")
            hypothesis, elapsed = _generate(llm, prompt, sampling, args.max_tokens)
            hypotheses.append(hypothesis)
            times.append(elapsed)
            print(f"{name}\t{elapsed:.2f}s\t{hypothesis[:120].replace(chr(10), ' ')}", flush=True)
        condition_report = {
            "sampling": {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
            },
            "scores": _scores(hypotheses, [reference for _source, reference in pairs]),
            "latency": _summarize(times),
            "hypotheses": hypotheses,
        }
        report["conditions"][name] = condition_report  # type: ignore[index]
        print(name, json.dumps({k: condition_report[k] for k in ("scores", "latency")}), flush=True)

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
