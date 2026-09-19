from agent.rag import chunk_markdown, dedupe

from .conftest import chunk


def test_chunks_carry_citable_heading_paths(tmp_path):
    doc = tmp_path / "runbook.md"
    doc.write_text(
        "# Runbook\n\nIntro.\n\n## Constraints\n\nNever below 768 MiB.\n\n"
        "### Replicas\n\nAt least 3.\n\n## Metrics\n\nBursty.\n"
    )
    chunks = chunk_markdown(doc, tmp_path)
    assert [c.citation for c in chunks] == [
        "runbook.md > Runbook",
        "runbook.md > Constraints",
        "runbook.md > Constraints > Replicas",
        "runbook.md > Metrics",
    ]
    # the address is embedded with the text so it influences similarity
    assert chunks[1].text.startswith("[runbook.md > Constraints]")
    assert len({c.id for c in chunks}) == len(chunks)


def test_dedupe_keeps_best_copy_in_distance_order():
    a_far = chunk("a.md", "x", distance=0.9)
    a_near = chunk("a.md", "x", distance=0.2)
    b = chunk("b.md", "y", distance=0.5)
    assert [(c.source, c.distance) for c in dedupe([a_far, b, a_near])] == [
        ("a.md", 0.2), ("b.md", 0.5)
    ]
