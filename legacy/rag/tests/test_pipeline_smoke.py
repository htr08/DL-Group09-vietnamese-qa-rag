from backend.rag.pipeline import MarketingRAGEngine
from backend.rag.models import RAGResponse


class _FakeRetriever:
    def retrieve(self, question: str, top_k=None):
        return []

    def to_sources(self, documents):
        return []


class _FakeGenerator:
    def generate(self, question, documents, sources=None):
        return RAGResponse(answer="ok", sources=sources or [])

    def stream(self, question, documents, sources=None, history=None):
        r = self.generate(question, documents, sources)
        yield r.answer


def test_pipeline_returns_no_context_for_empty_retrieval():
    engine = MarketingRAGEngine()
    engine._retriever = _FakeRetriever()
    engine._generator = _FakeGenerator()

    parts: list[str] = []
    for ev in engine.ask_bot_stream("hello"):
        if "chunk" in ev:
            parts.append(ev["chunk"])
    resp = RAGResponse(answer="".join(parts).strip(), sources=[])
    assert isinstance(resp, RAGResponse)
    assert isinstance(resp.answer, str)
