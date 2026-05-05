from __future__ import annotations

from .policy import EvidencePolicy
from .response_cache import ResponseCache
from .schemas import EvidenceRequest, EvidenceResponse


class EvidenceService:
    def __init__(self, *, policy: EvidencePolicy | None = None, cache: ResponseCache | None = None) -> None:
        self.policy = policy
        self.cache = cache or ResponseCache()

    async def run_request(self, request: EvidenceRequest, *, query_budget_remaining: int | None = None) -> EvidenceResponse:
        key = self.cache.make_key(request.question, request.intent, request.drug_names)
        cached = self.cache.get(key)
        if cached is not None:
            cached.meta.cached = True
            return cached
        policy = self.policy or EvidencePolicy()
        response = await policy.run(request, query_budget_remaining=query_budget_remaining)
        self.cache.set(key, response)
        return response

    async def run(
        self,
        *,
        question: str,
        intent: str,
        drug_names: list[str],
        query_budget_remaining: int | None = None,
    ) -> dict:
        request = EvidenceRequest(question=question, intent=intent, drug_names=drug_names)
        response = await self.run_request(request, query_budget_remaining=query_budget_remaining)
        return response.model_dump()


evidence_service = EvidenceService()
