from PIL import Image

from experiments.e4.tasks.finers_qa.utils import (
    answerable,
    finers_doc_to_text,
    finers_process_results,
)
from experiments.e4.tasks.hrscene.utils import hrscene_process_results
from experiments.e4.tasks.visualprobe.utils import (
    visualprobe_doc_to_visual,
    visualprobe_process_results,
)


def test_visualprobe_adapter_reads_image_and_normalizes_punctuation(tmp_path, monkeypatch):
    path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4)).save(path)
    monkeypatch.setenv("VISUALPROBE_ROOT", str(tmp_path))
    doc = {"images": ["sample.png"], "problem": "<image>\nColor?", "solution": "Blue."}
    assert visualprobe_doc_to_visual(doc)[0].size == (4, 4)
    assert visualprobe_process_results(doc, ["blue"])["visualprobe_accuracy"] == 1.0


def test_finers_adapter_excludes_referring_and_handles_mcq():
    referring = {"annotations": {"A": "", "Q": "tower"}}
    option = {
        "annotations": {
            "A": "B",
            "Q": "What color?",
            "options": "(A) red (B) white",
        }
    }
    assert not answerable(referring)
    assert answerable(option)
    assert "option letter" in finers_doc_to_text(option)
    assert finers_process_results(option, ["The final answer is B."])["finers_qa_accuracy"] == 1.0


def test_hrscene_adapter_is_case_and_punctuation_insensitive():
    assert hrscene_process_results({"answer": "New York."}, ["new york"])["hrscene_accuracy"] == 1.0
