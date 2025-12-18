import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so that `owg_mod` can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from owg_mod.prompt_library import SystemPromptLibrary  # noqa: E402


def test_system_prompt_library_loads_txt_files(tmp_path):
    # Create a temporary prompt directory with a few files
    (tmp_path / "referring_segmentation_base.txt").write_text(
        "Prompt base: {object_name}", encoding="utf-8"
    )
    (tmp_path / "referring_segmentation_confidence.txt").write_text(
        "Prompt confidence: {object_name} with confidence {confidence}", encoding="utf-8"
    )
    (tmp_path / "unrelated.md").write_text("not a prompt", encoding="utf-8")

    lib = SystemPromptLibrary(str(tmp_path))

    available = lib.list_available_prompts()
    # Only .txt files without extension should appear
    assert "referring_segmentation_base" in available
    assert "referring_segmentation_confidence" in available
    assert "unrelated" not in available


def test_system_prompt_library_load_and_prepare_prompt(tmp_path):
    (tmp_path / "test_prompt.txt").write_text("Hello, {name}!", encoding="utf-8")
    lib = SystemPromptLibrary(str(tmp_path))

    raw = lib.load_prompt("test_prompt")
    assert raw == "Hello, {name}!"

    filled = lib.prepare_prompt("test_prompt", {"name": "World"})
    assert filled == "Hello, World!"


def test_system_prompt_library_prepare_prompt_missing_raises(tmp_path):
    (tmp_path / "test_prompt.txt").write_text("Hello, {name}!", encoding="utf-8")
    lib = SystemPromptLibrary(str(tmp_path))

    with pytest.raises(ValueError):
        lib.prepare_prompt("nonexistent", {"name": "World"})

    # Missing variable should also raise ValueError
    with pytest.raises(ValueError):
        lib.prepare_prompt("test_prompt", {})


def test_system_prompt_library_prepare_variant_prompts(tmp_path, capsys):
    (tmp_path / "task_base.txt").write_text("Base: {x}", encoding="utf-8")
    (tmp_path / "task_confidence.txt").write_text("Confidence: {x}", encoding="utf-8")
    # Intentionally omit one variant to trigger error path
    lib = SystemPromptLibrary(str(tmp_path))

    variants = lib.prepare_variant_prompts(
        base_name="task", variants=["_base", "_confidence", "_missing"], variables={"x": 5}
    )

    # Only existing variants should be returned
    assert variants["_base"] == "Base: 5"
    assert variants["_confidence"] == "Confidence: 5"
    assert "_missing" not in variants

    # Error message for missing variant should have been printed
    captured = capsys.readouterr().out
    assert "Error loading variant" in captured


def test_system_prompt_library_read_prompt_from_file(tmp_path):
    filename = tmp_path / "direct_read.txt"
    filename.write_text("Raw content", encoding="utf-8")

    lib = SystemPromptLibrary(str(tmp_path))

    # read existing file
    content = lib.read_prompt_from_file("direct_read")
    assert content == "Raw content"

    # with explicit .txt
    content2 = lib.read_prompt_from_file("direct_read.txt")
    assert content2 == "Raw content"

    # non-existent file returns None
    assert lib.read_prompt_from_file("does_not_exist") is None



