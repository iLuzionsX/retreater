"""Prompt text, sampling, and AST parsing shared by MLX and the CPU eval harness.

The baseline builders reproduce the prompts that shipped with the imported tree.
The current builders are what the live worker uses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int


# mlx-vlm's generate() default. Faster on CPU, but lower BLEU on Gemma 4 E4B-it.
GREEDY_SAMPLING = SamplingConfig(temperature=0.0, top_p=1.0, top_k=0)

# Google's published Gemma 4 chat sampler. The CPU Q8_0 eval preferred this
# over greedy for the live translation prompt (higher BLEU and chrF).
GEMMA4_CHAT_SAMPLING = SamplingConfig(temperature=1.0, top_p=0.95, top_k=64)

# Live caption tasks use the sampler that won the off-Mac quality comparison.
CAPTION_SAMPLING = GEMMA4_CHAT_SAMPLING


def _clean(text: str) -> str:
    return " ".join(text.strip().split())


def format_bilingual_context(pairs: list[tuple[str, str]], *, limit: int = 5) -> str:
    lines: list[str] = []
    for original, translation in pairs[-limit:]:
        original = _clean(original)
        translation = _clean(translation)
        if not original or not translation:
            continue
        lines.append(f"- Source: {original}\n  Translation: {translation}")
    return "\n".join(lines)


def build_ast_prompt_baseline(
    src: str,
    tgt: str,
    *,
    prior_context: list[tuple[str, str]] | None = None,
    custom_vocab: list[str] | None = None,
    code_switching_enabled: bool = False,
) -> str:
    prompt_text = (
        f"Transcribe the following speech segment in {src}, "
        f"then translate it into {tgt}. When formatting the answer, "
        f"first output the transcription in {src}, then one newline, "
        f"then output the string '{tgt}: ', then the translation in {tgt}."
    )
    if custom_vocab:
        prompt_text = (
            f"The speaker frequently uses these terms: {', '.join(custom_vocab)}. "
            "Prefer them when acoustically ambiguous.\n\n" + prompt_text
        )
    if code_switching_enabled:
        prompt_text = (
            f"Speaker may code-switch between {src} and {tgt}; "
            f"transcribe in the spoken language, translate to {tgt}.\n\n" + prompt_text
        )
    if prior_context:
        ctx = "\n".join(f"Previous: {original} / {translation}" for original, translation in prior_context[-2:])
        prompt_text = ctx + "\n\n" + prompt_text
    return prompt_text


def build_ast_prompt(
    src: str,
    tgt: str,
    *,
    prior_context: list[tuple[str, str]] | None = None,
    custom_vocab: list[str] | None = None,
    code_switching_enabled: bool = False,
) -> str:
    """Instruction first, then references. A leading context block makes the model continue it."""
    parts = [
        (
            f"Transcribe this speech segment in {src}, then translate it into {tgt}. "
            "Return only two parts: the transcription, then a newline, then the line "
            f"'{tgt}: ' followed by the translation. Do not add a preamble, quotes, or notes."
        )
    ]
    if custom_vocab:
        terms = ", ".join(term.strip() for term in custom_vocab if term.strip())
        if terms:
            parts.append(
                f"The speaker frequently uses these terms: {terms}. "
                "Prefer them when the audio is ambiguous."
            )
    if code_switching_enabled:
        parts.append(
            f"The speaker may code-switch between {src} and {tgt}. "
            f"Transcribe each word in the language that was spoken, and translate the whole segment into {tgt}."
        )
    context = format_bilingual_context(prior_context or [], limit=2)
    if context:
        parts.append(
            "Recent committed captions. Use them only to keep names and repeated phrases consistent. "
            "Do not repeat them unless they were spoken again.\n"
            + context
        )
    return "\n\n".join(parts)


def build_translate_prompt_baseline(
    text: str,
    src: str,
    tgt: str,
    *,
    bilingual_context: list[tuple[str, str]] | None = None,
) -> str:
    context = format_bilingual_context(bilingual_context or [])
    if context:
        return (
            f"Translate the following text from {src} to {tgt}. Return ONLY the translation "
            "with no preamble.\n\n"
            "Use these recent bilingual reference pairs only to preserve names, recurring "
            "church terms, scripture wording, tone, and phrase choices when they clearly "
            "apply. Do not copy a reference pair unless it matches the text being translated.\n\n"
            f"{context}\n\nText: {text}"
        )
    return (
        f"Translate the following text from {src} to {tgt}. Return ONLY the translation "
        f"with no preamble.\n\nText: {text}"
    )


def build_translate_prompt(
    text: str,
    src: str,
    tgt: str,
    *,
    bilingual_context: list[tuple[str, str]] | None = None,
) -> str:
    parts = [
        (
            f"Translate the text from {src} to {tgt}. "
            f"Return only the {tgt} translation, with the same meaning, names, and numbers. "
            "No preamble, labels, or quotation marks."
        )
    ]
    context = format_bilingual_context(bilingual_context or [])
    if context:
        parts.append(
            "Recent bilingual captions. Reuse a name or phrase only when it clearly refers to the same thing. "
            "Do not copy a reference unless it matches this text.\n" + context
        )
    parts.append(f"Text:\n{text.strip()}")
    return "\n\n".join(parts)


def build_polish_prompt(text: str) -> str:
    return (
        "Clean this rough transcription. Remove filler words (um, uh, er, you know, like), "
        "fix punctuation and capitalization, and keep the exact meaning and wording. "
        "Return only the cleaned text.\n\n"
        f"Transcription:\n{text.strip()}"
    )


def build_polish_prompt_baseline(text: str) -> str:
    return (
        "You will receive a rough transcription. Remove filler words "
        "(um, uh, er, you know, like), fix punctuation, fix capitalization, "
        "and keep the exact meaning and wording. Return ONLY the cleaned text "
        f"with no preamble.\n\nTranscription: {text}"
    )


def build_asr_correction_prompt(
    text: str,
    src: str,
    *,
    custom_vocab: list[str] | None = None,
    prior_context: list[tuple[str, str]] | None = None,
    learned_corrections: list[tuple[str, str]] | None = None,
    code_switching_enabled: bool = False,
) -> str:
    context_parts: list[str] = []
    if custom_vocab:
        terms = ", ".join(item.strip() for item in custom_vocab if item.strip())
        if terms:
            context_parts.append(f"Known names, terms, or phrases: {terms}.")
    if prior_context:
        recent = "\n".join(f"- {original}" for original, _translation in prior_context[-3:] if original.strip())
        if recent:
            context_parts.append("Recent transcript context:\n" + recent)
    learned_lines: list[str] = []
    for heard, corrected in (learned_corrections or [])[-8:]:
        heard = _clean(heard)
        corrected = _clean(corrected)
        if not heard or not corrected or heard == corrected:
            continue
        learned_lines.append(f"- Heard: {heard}\n  Corrected: {corrected}")
    if learned_lines:
        context_parts.append(
            "Previously corrected recognition mistakes. Prefer the corrected wording when it matches:\n"
            + "\n".join(learned_lines)
        )
    if code_switching_enabled:
        context_parts.append(f"The speaker may code-switch while primarily speaking {src}.")
    context = "\n\n".join(context_parts)
    if context:
        context += "\n\n"
    return (
        f"You will receive a finalized ASR transcript in {src}. Correct likely speech "
        "recognition errors, punctuation, capitalization, and spacing. Preserve the "
        "speaker's wording and meaning. Do not summarize, translate, add commentary, "
        "or censor. Never translate the transcript; if the transcript is actually in "
        "another language, keep that language and only fix recognition mistakes. "
        "Use the optional context only when it plausibly matches what was "
        f"said. Return ONLY the corrected {src} transcript.\n\n{context}Transcript: {text}"
    )


def parse_ast_output(text: str, target_lang: str) -> tuple[str, str]:
    """Split an AST completion into source text and translation.

    The model is asked for ``<source>\\n{target}: <translation>``. It also emits
    the marker without the leading newline, with a different colon, or after a
    short preamble. Those forms used to drop the translation.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

    marker = _find_target_marker(cleaned, target_lang)
    if marker is None:
        return _strip_wrapping_quotes(cleaned), ""
    start, end = marker
    original = _strip_wrapping_quotes(cleaned[:start].strip())
    translation = _strip_wrapping_quotes(cleaned[end:].strip())
    return original, translation


def _find_target_marker(text: str, target_lang: str) -> tuple[int, int] | None:
    label = target_lang.strip()
    if not label:
        return None
    lowered = text.lower()
    label_lower = label.lower()
    search_from = 0
    while True:
        index = lowered.find(label_lower, search_from)
        if index < 0:
            return None
        end = index + len(label_lower)
        tail = text[end:]
        if tail.startswith(":") or tail.startswith("："):
            prefix = text[:index]
            if index == 0 or prefix.endswith(("\n", " ")) or prefix.rstrip().endswith("\n"):
                return index, end + 1
        search_from = index + 1


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1].strip()
    return stripped
