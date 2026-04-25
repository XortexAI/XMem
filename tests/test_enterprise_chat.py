import os

import pytest

os.environ.setdefault("PINECONE_API_KEY", "test-pinecone-key")
os.environ.setdefault("NEO4J_PASSWORD", "test-neo4j-password")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from src.enterprise.annotation_service import EnterpriseAnnotationService
from src.enterprise.chat_orchestrator import EnterpriseChatOrchestrator
from src.enterprise.schemas import (
    EnterpriseChatContext,
    EnterpriseChatRequest,
    EnterpriseMemoryContext,
    EnterpriseProjectContext,
    EnterpriseUserContext,
)
from src.schemas.code import (
    AnnotationSeverity,
    AnnotationType,
    CodeAnnotationResult,
    ExtractedAnnotation,
)


def _context(can_create_annotations=True, role="manager"):
    return EnterpriseChatContext(
        project=EnterpriseProjectContext(
            project_id="proj_1",
            org_id="acme",
            repo="payments",
        ),
        user=EnterpriseUserContext(
            user_id="user_1",
            username="alice",
            role=role,
            can_create_annotations=can_create_annotations,
        ),
        request=EnterpriseChatRequest(
            query="The retry path can duplicate charges.",
            file_path="src/payments/processor.py",
            symbol_name="PaymentProcessor.process",
            top_k=7,
        ),
    )


class FakeAnnotationStore:
    def __init__(self):
        self.created = []

    def create_annotation(self, **kwargs):
        self.created.append(kwargs)
        return f"ann_{len(self.created)}"


class FakeCodeAgent:
    def __init__(self, result):
        self.result = result
        self.states = []

    async def arun(self, state):
        self.states.append(state)
        return self.result


@pytest.mark.asyncio
async def test_annotation_service_extracts_and_stores_project_annotations():
    store = FakeAnnotationStore()
    count_updates = []
    agent = FakeCodeAgent(
        CodeAnnotationResult(
            annotations=[
                ExtractedAnnotation(
                    target_symbol="PaymentProcessor.process",
                    target_file="src/payments/processor.py",
                    content="Retry path can duplicate charges without idempotency.",
                    annotation_type=AnnotationType.BUG_REPORT,
                    severity=AnnotationSeverity.HIGH,
                    repo="payments",
                )
            ]
        )
    )
    service = EnterpriseAnnotationService(
        annotation_store=store,
        code_agent_factory=lambda: agent,
        annotation_count_incrementer=lambda project_id, count: count_updates.append(
            (project_id, count)
        ),
    )

    ids = await service.extract_and_store(
        context=_context(),
        answer_text="Use an idempotency key per transaction.",
    )

    assert ids == ["ann_1"]
    assert len(store.created) == 1
    created = store.created[0]
    assert created["project_id"] == "proj_1"
    assert created["org_id"] == "acme"
    assert created["repo"] == "payments"
    assert created["author_id"] == "user_1"
    assert created["author_role"] == "manager"
    assert created["annotation_type"] == "bug_report"
    assert created["severity"] == "high"
    assert created["file_path"] == "src/payments/processor.py"
    assert created["symbol_name"] == "PaymentProcessor.process"
    assert count_updates == [("proj_1", 1)]
    assert "Assistant response" in agent.states[0]["classifier_output"]


@pytest.mark.asyncio
async def test_annotation_service_skips_users_without_annotation_permission():
    store = FakeAnnotationStore()

    class RaisingAgent:
        async def arun(self, state):
            raise AssertionError("agent should not run")

    service = EnterpriseAnnotationService(
        annotation_store=store,
        code_agent_factory=RaisingAgent,
    )

    ids = await service.extract_and_store(
        context=_context(can_create_annotations=False, role="intern"),
        answer_text="Ignored.",
    )

    assert ids == []
    assert store.created == []


class FakeTeamAnnotationStore:
    async def search_relevant_for_query(self, **kwargs):
        return [
            {
                "id": "ann_existing",
                "content": "Always preserve idempotency in payment retries.",
                "annotation_type": "warning",
                "author_name": "Priya",
                "author_role": "manager",
            }
        ]

    async def get_manager_instructions(self, **kwargs):
        return []


class FakeMemoryService:
    def __init__(self):
        self.ingested = []

    async def fetch_relevant_memory(self, **kwargs):
        return EnterpriseMemoryContext(
            sources=[
                {
                    "domain": "summary",
                    "content": "Alice has been debugging payment retries.",
                    "score": 0.9,
                    "metadata": {},
                }
            ]
        )

    async def ingest_conversation(self, query, answer_text, user_id):
        self.ingested.append((query, answer_text, user_id))

        class Result:
            success = True
            error = None

        return Result()


class FakeEnterpriseAnnotationService:
    def __init__(self):
        self.calls = []

    async def extract_and_store(self, context, answer_text):
        self.calls.append((context, answer_text))
        return ["ann_created"]


class FakeCodePipeline:
    def __init__(self):
        self.calls = []

    async def run_stream(self, query, user_id, repo, top_k):
        self.calls.append({
            "query": query,
            "user_id": user_id,
            "repo": repo,
            "top_k": top_k,
        })
        yield '{"type": "chunk", "text": "Use idempotency keys."}\n'
        yield '{"type": "done"}\n'


@pytest.mark.asyncio
async def test_orchestrator_unifies_annotations_memory_and_code_pipeline():
    pipeline = FakeCodePipeline()
    memory_service = FakeMemoryService()
    annotation_service = FakeEnterpriseAnnotationService()
    pipeline_args = []

    def code_pipeline_factory(org_id, repo, project_id):
        pipeline_args.append((org_id, repo, project_id))
        return pipeline

    orchestrator = EnterpriseChatOrchestrator(
        annotation_store=FakeTeamAnnotationStore(),
        annotation_service=annotation_service,
        memory_service=memory_service,
        code_pipeline_factory=code_pipeline_factory,
    )

    chunks = [
        chunk
        async for chunk in orchestrator.stream_chat(_context())
    ]

    assert pipeline_args == [("acme", "payments", "proj_1")]
    call = pipeline.calls[0]
    assert call["user_id"] == "alice"
    assert call["repo"] == "payments"
    assert call["top_k"] == 7
    assert "Team Knowledge" in call["query"]
    assert "User Memory" in call["query"]
    assert any('"type": "annotations"' in chunk for chunk in chunks)
    assert any('"type": "memory_sources"' in chunk for chunk in chunks)
    assert any('"type": "annotations_created"' in chunk for chunk in chunks)
    assert annotation_service.calls[0][1] == "Use idempotency keys."
    assert memory_service.ingested == [
        (
            "The retry path can duplicate charges.",
            "Use idempotency keys.",
            "alice",
        )
    ]
