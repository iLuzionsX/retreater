from prompts import (
    build_ast_prompt,
    build_ast_prompt_baseline,
    build_translate_prompt,
    build_translate_prompt_baseline,
    parse_ast_output,
)


def test_parse_ast_output_accepts_newline_and_inline_markers() -> None:
    original, translation = parse_ast_output("Good morning.\nSpanish: Buenos días.", "Spanish")
    assert original == "Good morning."
    assert translation == "Buenos días."

    original, translation = parse_ast_output("Good morning. Spanish: Buenos días.", "Spanish")
    assert original == "Good morning."
    assert translation == "Buenos días."


def test_parse_ast_output_keeps_source_when_marker_is_missing() -> None:
    original, translation = parse_ast_output("Just the transcript", "Spanish")
    assert original == "Just the transcript"
    assert translation == ""


def test_ast_prompt_puts_context_after_the_instruction() -> None:
    prompt = build_ast_prompt(
        "English",
        "Spanish",
        prior_context=[("Grace", "Gracia")],
    )
    instruction_at = prompt.index("Transcribe this speech")
    context_at = prompt.index("Source: Grace")
    assert instruction_at < context_at

    baseline = build_ast_prompt_baseline(
        "English",
        "Spanish",
        prior_context=[("Grace", "Gracia")],
    )
    assert baseline.startswith("Previous: Grace / Gracia")


def test_translate_prompt_asks_for_translation_only() -> None:
    prompt = build_translate_prompt("Peace be with you.", "English", "Spanish")
    assert "Peace be with you." in prompt
    assert "only the Spanish translation" in prompt
    baseline = build_translate_prompt_baseline("Peace be with you.", "English", "Spanish")
    assert baseline.startswith("Translate the following text")
