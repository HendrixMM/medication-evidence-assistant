#!/usr/bin/env python3
"""
Unit tests for citation formatting with missing metadata.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from synthesis_engine import SynthesisEngine


def test_citation_formatting():
    """Test citation formatting with various missing metadata scenarios."""
    print("Testing citation formatting...")

    # Create synthesis engine instance
    engine = SynthesisEngine()

    # Test case 1: Complete metadata
    papers_complete = [
        {
            "metadata": {
                "title": "Complete Study",
                "authors": ["Smith, J.", "Johnson, A.", "Brown, K."],
                "journal": "Journal of Medical Research",
                "year": "2023",
                "pmid": "12345678",
            }
        }
    ]

    citations = engine._format_citations(papers_complete)
    print("Complete metadata citation:", citations[0])
    assert "Smith, J." in citations[0]
    assert "Complete Study" in citations[0]
    assert "Journal of Medical Research" in citations[0]
    assert "2023" in citations[0]
    assert "PMID: 12345678" in citations[0]

    # Test case 2: Missing PMID but has DOI
    papers_doi = [
        {
            "metadata": {
                "title": "DOI Study",
                "authors": ["Doe, J.", "Lee, M."],
                "journal": "Clinical Pharmacology",
                "year": "2022",
                "doi": "10.1234/56789",
            }
        }
    ]

    citations = engine._format_citations(papers_doi)
    print("DOI citation:", citations[0])
    assert "Doe, J." in citations[0]
    assert "DOI Study" in citations[0]
    assert "Clinical Pharmacology" in citations[0]
    assert "2022" in citations[0]
    assert "DOI: 10.1234/56789" in citations[0]

    # Test case 3: Missing both PMID and DOI but has URL
    papers_url = [
        {
            "metadata": {
                "title": "URL Study",
                "authors": ["Wilson, R."],
                "journal": "Research Notes",
                "year": "2021",
                "source_url": "https://example.com/study",
            }
        }
    ]

    citations = engine._format_citations(papers_url)
    print("URL citation:", citations[0])
    assert "Wilson, R." in citations[0]
    assert "URL Study" in citations[0]
    assert "Research Notes" in citations[0]
    assert "2021" in citations[0]
    assert "URL: https://example.com/study" in citations[0]

    # Test case 4: Missing title
    papers_no_title = [
        {"metadata": {"authors": ["Taylor, S."], "journal": "Medical Journal", "year": "2020", "pmid": "87654321"}}
    ]

    citations = engine._format_citations(papers_no_title)
    print("No title citation:", citations[0])
    assert "Taylor, S." in citations[0]
    assert "Untitled Paper" in citations[0]  # Should use fallback
    assert "Medical Journal" in citations[0]
    assert "2020" in citations[0]
    assert "PMID: 87654321" in citations[0]

    # Test case 5: Missing authors
    papers_no_authors = [
        {
            "metadata": {
                "title": "No Authors Study",
                "journal": "Science Reports",
                "year": "2019",
                "doi": "10.5678/90123",
            }
        }
    ]

    citations = engine._format_citations(papers_no_authors)
    print("No authors citation:", citations[0])
    assert "Unknown Authors" in citations[0]
    assert "No Authors Study" in citations[0]
    assert "Science Reports" in citations[0]
    assert "2019" in citations[0]
    assert "DOI: 10.5678/90123" in citations[0]

    # Test case 6: Missing year
    papers_no_year = [
        {
            "metadata": {
                "title": "No Year Study",
                "authors": ["Miller, P.", "Davis, T."],
                "journal": "Health Research",
                "pmid": "11223344",
            }
        }
    ]

    citations = engine._format_citations(papers_no_year)
    print("No year citation:", citations[0])
    assert "Miller, P." in citations[0]
    assert "No Year Study" in citations[0]
    assert "Health Research" in citations[0]
    assert "Unknown Year" in citations[0]
    assert "PMID: 11223344" in citations[0]

    # Test case 7: Missing journal
    papers_no_journal = [
        {"metadata": {"title": "No Journal Study", "authors": ["Anderson, L."], "year": "2018", "doi": "10.9876/54321"}}
    ]

    citations = engine._format_citations(papers_no_journal)
    print("No journal citation:", citations[0])
    assert "Anderson, L." in citations[0]
    assert "No Journal Study" in citations[0]
    assert "Unknown Journal" in citations[0]
    assert "2018" in citations[0]
    assert "DOI: 10.9876/54321" in citations[0]

    # Test case 8: Missing everything except URL
    papers_minimal = [{"metadata": {"source_url": "https://minimal.example.com"}}]

    citations = engine._format_citations(papers_minimal)
    print("Minimal metadata citation:", citations[0])
    assert "Unknown Authors" in citations[0]
    assert "Untitled Paper" in citations[0]
    assert "Unknown Journal" in citations[0]
    assert "Unknown Year" in citations[0]
    assert "URL: https://minimal.example.com" in citations[0]

    # Test case 9: Missing PMID, DOI, and URL
    papers_no_identifiers = [
        {
            "metadata": {
                "title": "No Identifiers Study",
                "authors": ["White, G.", "Black, H."],
                "journal": "Basic Research",
                "year": "2017",
            }
        }
    ]

    citations = engine._format_citations(papers_no_identifiers)
    print("No identifiers citation:", citations[0])
    assert "White, G." in citations[0]
    assert "No Identifiers Study" in citations[0]
    assert "Basic Research" in citations[0]
    assert "2017" in citations[0]
    # Should not have PMID, DOI, or URL
    assert "PMID:" not in citations[0]
    assert "DOI:" not in citations[0]
    assert "URL:" not in citations[0]

    print("✅ All citation formatting tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_citation_formatting()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
