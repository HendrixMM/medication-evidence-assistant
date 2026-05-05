import sys


def mock_requests_get_eutils(payloads_by_endpoint):
    class Resp:
        def __init__(self, status=200, text="", json_data=None):
            self.status_code = status
            self._text = text
            self._json = json_data

        def raise_for_status(self):
            if not (200 <= self.status_code < 300):
                raise RuntimeError(f"HTTP {self.status_code}")

        @property
        def text(self):
            return self._text

        def json(self):
            return self._json

    def _get(url, params=None, timeout=None):
        if "esearch.fcgi" in url:
            data = payloads_by_endpoint.get("esearch")
            return Resp(json_data=data)
        if "efetch.fcgi" in url:
            data = payloads_by_endpoint.get("efetch_xml", "")
            return Resp(text=data)
        if "elink.fcgi" in url:
            data = payloads_by_endpoint.get("elink", {"linksets": []})
            return Resp(json_data=data)
        return Resp(status=404)

    return _get


def test_eutils_mapping(monkeypatch):
    # Arrange minimal E-utilities responses
    esearch = {"esearchresult": {"idlist": ["12345678"]}}
    # XML with basic fields
    efetch_xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345678</PMID>
          <Article>
            <ArticleTitle>Sample Title</ArticleTitle>
            <Abstract>
              <AbstractText Label="Background">This is background.</AbstractText>
              <AbstractText>And abstract body.</AbstractText>
            </Abstract>
            <AuthorList>
              <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
              <Author><CollectiveName>Consortium X</CollectiveName></Author>
            </AuthorList>
            <Journal>
              <JournalIssue>
                <PubDate>
                  <Year>2024</Year><Month>May</Month><Day>01</Day>
                </PubDate>
              </JournalIssue>
              <Title>Journal of Tests</Title>
            </Journal>
          </Article>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="doi">10.1000/test.doi</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """.strip()
    elink = {"linksets": [{"linksetdbs": [{"dbto": "pmc", "links": [{"id": "PMC12345"}]}]}]}

    # Mock requests.get in eutils client
    sys.path.insert(0, str((__import__("pathlib").Path("src")).resolve()))
    import pubmed_eutils_client as eutils

    monkeypatch.setattr(
        eutils.requests,
        "get",
        mock_requests_get_eutils(
            {
                "esearch": esearch,
                "efetch_xml": efetch_xml,
                "elink": elink,
            }
        ),
    )

    client = eutils.PubMedEutilsClient(email="test@example.com")
    results = client.search_and_fetch("glioblastoma", max_items=1)
    assert results and results[0]["pmid"] == "12345678"
    assert results[0]["title"] == "Sample Title"
    assert "Background:" in results[0]["abstract"] and "abstract body" in results[0]["abstract"]
    assert results[0]["authors"].startswith("Smith, John") and "Consortium X" in results[0]["authors"]
    assert results[0]["doi"] == "10.1000/test.doi"
    assert results[0]["journal"] == "Journal of Tests"
    assert results[0]["full_text_url"].startswith("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/")
    assert results[0]["provider"] == "pubmed_eutils"

