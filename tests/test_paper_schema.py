"""Unit tests for the Paper schema."""
from src.paper_schema import Paper


def test_paper_schema_normalises_metadata() -> None:
    raw = {
        "page_content": "Study content on drug interactions.",
        "metadata": {
            "title": "Important DDI Study",
            "pmid": "123456",
            "pk_parameters": {"auc": {"fold_change": 2.1}},
        },
        "abstract": "Abstract text",
    }

    paper = Paper.model_validate(raw)
    assert paper.page_content == raw["page_content"]
    assert paper.title == "Important DDI Study"
    assert paper.pmid == "123456"
    assert paper.pk_parameters["auc"]["fold_change"] == 2.1

    prepared = paper.as_dict()
    assert prepared["metadata"]["title"] == "Important DDI Study"
    assert prepared["metadata"]["pmid"] == "123456"
    assert prepared["metadata"]["pk_parameters"]["auc"]["fold_change"] == 2.1
    assert prepared["__paper_schema_validated__"] is True


def test_paper_schema_uses_content_when_page_content_missing() -> None:
    raw = {
        "content": "Fallback content for the paper.",
        "metadata": {},
    }

    paper = Paper.model_validate(raw)
    assert paper.page_content == "Fallback content for the paper."

    prepared = paper.as_dict()
    assert prepared["page_content"] == "Fallback content for the paper."
