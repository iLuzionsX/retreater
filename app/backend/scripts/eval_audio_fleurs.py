"""Score Gemma 4 E4B audio-to-Spanish on a small public FLEURS subset.

Uses English speech from google/fleurs (en_us) and the Spanish sentence with the
same id (es_419). No gated credentials. Keeps the download to ``--limit`` clips.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mlx_worker import MLXWorker  # noqa: E402
from prompts import parse_ast_output  # noqa: E402


def _download(filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="google/fleurs", repo_type="dataset", filename=filename)


def _tsv_map(path: str) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        sentence_id, wav_name, text = parts[0], parts[1], parts[2]
        rows[sentence_id] = (wav_name, text)
    return rows


def load_pairs(limit: int) -> list[dict]:
    import io

    import pyarrow.parquet as pq
    import soundfile as sf

    english_rows = _tsv_map(_download("data/en_us/test.tsv"))
    spanish_rows = _tsv_map(_download("data/es_419/test.tsv"))
    shared_ids = [sentence_id for sentence_id in english_rows if sentence_id in spanish_rows]
    if len(shared_ids) < limit:
        raise RuntimeError(f"Only {len(shared_ids)} aligned FLEURS sentence ids")
    chosen = shared_ids[:limit]
    wanted = {english_rows[sentence_id][0]: sentence_id for sentence_id in chosen}
    english = pq.read_table(
        _download("parquet-data/en_us/test-00000-of-00001.parquet"),
        columns=["path", "audio"],
    )
    audio_by_name: dict[str, dict] = {}
    for path, audio in zip(english.column("path").to_pylist(), english.column("audio").to_pylist(), strict=True):
        name = Path(str(path)).name
        if name in wanted and isinstance(audio, dict):
            audio_by_name[name] = audio
    pairs: list[dict] = []
    for sentence_id in chosen:
        wav_name, english_text = english_rows[sentence_id]
        audio = audio_by_name.get(wav_name)
        if not audio or not audio.get("bytes"):
            continue
        samples, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        pairs.append(
            {
                "id": sentence_id,
                "english": english_text,
                "spanish": spanish_rows[sentence_id][1],
                "array": samples,
                "sampling_rate": int(sample_rate),
            }
        )
    if len(pairs) < limit:
        raise RuntimeError(f"Only loaded {len(pairs)} FLEURS clips with audio")
    return pairs


def _resample(audio, source_rate: int):
    import numpy as np

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if source_rate == 16_000:
        return samples
    duration = samples.shape[0] / source_rate
    target_length = max(1, int(round(duration * 16_000)))
    source_positions = np.linspace(0.0, 1.0, num=samples.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("scripts/mac-measure/fleurs-eval.json"))
    args = parser.parse_args()

    pairs = load_pairs(args.limit)
    worker = MLXWorker()
    hypotheses: list[str] = []
    references: list[str] = []
    rows: list[dict] = []
    for pair in pairs:
        audio = _resample(pair["array"], pair["sampling_rate"])
        started = time.perf_counter()
        original, translation = worker.ast(
            audio,
            "English",
            "Spanish",
            prior_context=[],
            max_tokens=128,
        )
        elapsed = time.perf_counter() - started
        hypotheses.append(translation)
        references.append(pair["spanish"])
        rows.append(
            {
                "id": pair["id"],
                "english": pair["english"],
                "reference": pair["spanish"],
                "hypothesis_original": original,
                "hypothesis": translation,
                "seconds": elapsed,
            }
        )
        print(f"{pair['id']}\t{elapsed:.2f}s\t{translation}")

    import sacrebleu

    bleu = float(sacrebleu.corpus_bleu(hypotheses, [references]).score)
    chrf = float(sacrebleu.corpus_chrf(hypotheses, [references]).score)
    summary = {
        "set": "google/fleurs test, en_us audio with es_419 text, matched on sentence id",
        "limit": args.limit,
        "bleu": bleu,
        "chrf": chrf,
        "median_seconds": statistics.median(row["seconds"] for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("bleu", "chrf", "median_seconds", "limit")}, indent=2))


if __name__ == "__main__":
    main()
