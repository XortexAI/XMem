"""Team annotation store using Pinecone for vector storage and retrieval.

Team annotations are stored in Pinecone with metadata for semantic search.
Each annotation is associated with a project and can target specific files/symbols.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.config import settings
from src.pipelines.ingest import embed_text
from src.storage.pinecone import PineconeVectorStore, SearchResult

logger = logging.getLogger("xmem.storage.team_annotation_store")


class TeamAnnotationStore:
    """Store for team annotations in Pinecone with project-scoped namespaces.

    Schema:
        namespace: "annotations:{project_id}"
        metadata: {
            "content": str,              # The annotation text
            "project_id": str,           # Project ID
            "org_id": str,               # GitHub org
            "repo": str,                 # Repository name
            "file_path": Optional[str],  # Target file (if any)
            "symbol_name": Optional[str],# Target symbol (if any)
            "author_id": str,            # User ID who created
            "author_name": str,          # Display name
            "author_role": str,          # TeamRole value
            "annotation_type": str,      # bug_report|fix|explanation|warning|feature_idea
            "severity": Optional[str],   # low|medium|high|critical
            "status": str,               # active|resolved|outdated
            "created_at": str,           # ISO timestamp
            "updated_at": str,           # ISO timestamp
        }
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        """Initialize the team annotation store.

        Args:
            api_key: Pinecone API key (falls back to settings)
            index_name: Pinecone index name (falls back to settings)
            dimension: Embedding dimension (falls back to settings)
        """
        self._api_key = api_key or settings.pinecone_api_key
        self._index_name = index_name or settings.pinecone_index_name
        self._dimension = dimension or settings.pinecone_dimension

        # Cache of project stores (namespace -> PineconeVectorStore)
        self._stores: Dict[str, PineconeVectorStore] = {}

    def _get_store(self, project_id: str) -> PineconeVectorStore:
        """Get or create a PineconeVectorStore for a project namespace."""
        namespace = f"annotations:{project_id}"

        if namespace not in self._stores:
            self._stores[namespace] = PineconeVectorStore(
                api_key=self._api_key,
                index_name=self._index_name,
                dimension=self._dimension,
                namespace=namespace,
                create_if_not_exists=True,
            )
            logger.debug(f"Created store for namespace: {namespace}")

        return self._stores[namespace]

    def create_annotation(
        self,
        project_id: str,
        content: str,
        author_id: str,
        author_name: str,
        author_role: str,
        org_id: str,
        repo: str,
        annotation_type: str = "explanation",
        file_path: Optional[str] = None,
        symbol_name: Optional[str] = None,
        severity: Optional[str] = None,
        status: str = "active",
        source_message_id: Optional[str] = None,
    ) -> str:
        """Create a new team annotation.

        Args:
            project_id: The project ID
            content: The annotation text content
            author_id: User ID of the author
            author_name: Display name of the author
            author_role: Role of the author (manager|staff_engineer|sde2|intern)
            org_id: GitHub organization ID
            repo: Repository name
            annotation_type: Type of annotation (bug_report|fix|explanation|warning|feature_idea)
            file_path: Optional target file path
            symbol_name: Optional target symbol name
            severity: Optional severity (low|medium|high|critical)
            status: Status (active|resolved|outdated)
            source_message_id: Optional chat message ID that created this

        Returns:
            The annotation ID
        """
        annotation_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        # Build metadata
        metadata = {
            "project_id": project_id,
            "org_id": org_id,
            "repo": repo,
            "author_id": author_id,
            "author_name": author_name,
            "author_role": author_role,
            "annotation_type": annotation_type,
            "status": status,
            "created_at": now,
            "updated_at": now,
        }

        # Add optional fields
        if file_path:
            metadata["file_path"] = file_path
        if symbol_name:
            metadata["symbol_name"] = symbol_name
        if severity:
            metadata["severity"] = severity
        if source_message_id:
            metadata["source_message_id"] = source_message_id

        # Embed the content
        embedding = embed_text(content)

        # Get project store and upsert
        store = self._get_store(project_id)
        store.add(
            texts=[content],
            embeddings=[embedding],
            ids=[annotation_id],
            metadata=[metadata],
        )

        logger.info(
            f"Created annotation {annotation_id} for project {project_id} "
            f"targeting {symbol_name or file_path or repo}"
        )

        return annotation_id

    async def search_annotations(
        self,
        project_id: str,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search annotations by semantic similarity.

        Args:
            project_id: The project ID to search within
            query: The query text to search for
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"status": "active"})

        Returns:
            List of SearchResult objects
        """
        store = self._get_store(project_id)

        # Add default status filter if not specified
        if filters is None:
            filters = {}
        if "status" not in filters:
            filters["status"] = "active"

        return await store.search_by_text(
            query_text=query,
            top_k=top_k,
            filters=filters,
        )

    def get_annotations_for_file(
        self,
        project_id: str,
        file_path: str,
        top_k: int = 50,
    ) -> List[SearchResult]:
        """Get all annotations targeting a specific file.

        Args:
            project_id: The project ID
            file_path: The file path
            top_k: Max annotations to return

        Returns:
            List of SearchResult objects
        """
        store = self._get_store(project_id)

        # Use metadata-only search with file_path filter
        return store.search_by_metadata(
            filters={
                "file_path": file_path,
                "status": "active",
            },
            top_k=top_k,
        )

    def get_annotations_for_symbol(
        self,
        project_id: str,
        symbol_name: str,
        file_path: Optional[str] = None,
        top_k: int = 50,
    ) -> List[SearchResult]:
        """Get all annotations targeting a specific symbol.

        Args:
            project_id: The project ID
            symbol_name: The fully qualified symbol name
            file_path: Optional file path to narrow results
            top_k: Max annotations to return

        Returns:
            List of SearchResult objects
        """
        store = self._get_store(project_id)

        filters = {
            "symbol_name": symbol_name,
            "status": "active",
        }
        if file_path:
            filters["file_path"] = file_path

        return store.search_by_metadata(
            filters=filters,
            top_k=top_k,
        )

    def update_annotation(
        self,
        project_id: str,
        annotation_id: str,
        content: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> bool:
        """Update an existing annotation.

        Args:
            project_id: The project ID
            annotation_id: The annotation ID
            content: New content (optional)
            status: New status (optional)
            severity: New severity (optional)

        Returns:
            True if updated successfully
        """
        store = self._get_store(project_id)

        # Build metadata updates
        metadata = {"updated_at": datetime.utcnow().isoformat()}
        if status:
            metadata["status"] = status
        if severity:
            metadata["severity"] = severity

        # Get new embedding if content changed
        embedding = None
        if content:
            embedding = embed_text(content)

        return store.update(
            id=annotation_id,
            text=content,
            embedding=embedding,
            metadata=metadata,
        )

    def delete_annotation(
        self,
        project_id: str,
        annotation_id: str,
    ) -> bool:
        """Delete an annotation.

        Args:
            project_id: The project ID
            annotation_id: The annotation ID

        Returns:
            True if deleted successfully
        """
        store = self._get_store(project_id)
        return store.delete(ids=[annotation_id])

    def get_annotation(
        self,
        project_id: str,
        annotation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific annotation by ID.

        Args:
            project_id: The project ID
            annotation_id: The annotation ID

        Returns:
            The annotation dict or None if not found
        """
        store = self._get_store(project_id)
        results = store.get(ids=[annotation_id])

        if results:
            return results[0]
        return None

    def count_annotations(
        self,
        project_id: str,
    ) -> int:
        """Count annotations in a project.

        Args:
            project_id: The project ID

        Returns:
            Number of annotations
        """
        store = self._get_store(project_id)
        return store.count()

    async def search_relevant_for_query(
        self,
        project_id: str,
        query: str,
        file_path: Optional[str] = None,
        symbol_name: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for annotations relevant to a code query.

        This is used during code chat to find relevant team annotations
        based on the user's query and optionally a specific file/symbol.

        Args:
            project_id: The project ID
            query: The user's query text
            file_path: Optional file path to prioritize
            symbol_name: Optional symbol name to prioritize
            top_k: Number of results to return

        Returns:
            List of annotation dicts with relevance scores
        """
        # First do a semantic search
        results = await self.search_annotations(
            project_id=project_id,
            query=query,
            top_k=top_k * 2,  # Get more to filter
        )

        # If we have a specific file/symbol, also get those annotations
        if file_path or symbol_name:
            specific_results = []
            if symbol_name:
                specific_results.extend(
                    self.get_annotations_for_symbol(
                        project_id=project_id,
                        symbol_name=symbol_name,
                        file_path=file_path,
                        top_k=10,
                    )
                )
            elif file_path:
                specific_results.extend(
                    self.get_annotations_for_file(
                        project_id=project_id,
                        file_path=file_path,
                        top_k=10,
                    )
                )

            # Merge results, prioritizing specific matches
            seen_ids = {r.id for r in results}
            for r in specific_results:
                if r.id not in seen_ids:
                    results.append(r)

        # Convert to dicts and return top_k
        annotations = []
        for r in results[:top_k]:
            ann = {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                **r.metadata,
            }
            annotations.append(ann)

        return annotations

    async def get_manager_instructions(
        self,
        project_id: str,
        target_role: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get manager instructions for team members.

        This retrieves annotations created by managers that are marked as
        instructions or general guidance for the team.

        Args:
            project_id: The project ID
            target_role: Optional target role to filter for (e.g., 'intern', 'sde2')
            top_k: Maximum number of instructions to return

        Returns:
            List of manager instruction annotation dicts
        """
        store = self._get_store(project_id)

        # Build filters for manager annotations
        filters = {
            "status": "active",
            "author_role": "manager",
        }

        # Search for all manager annotations
        # We use a broad query to get instructions
        results = await store.search_by_text(
            query_text="instruction guidance manager note important",
            top_k=top_k,
            filters=filters,
        )

        instructions = []
        for r in results:
            meta = r.metadata or {}
            # Include annotations that are instructions or general explanations
            ann_type = meta.get("annotation_type", "explanation")
            if ann_type in ("instruction", "explanation", "warning", "feature_idea"):
                instruction = {
                    "id": r.id,
                    "content": r.content,
                    "author_name": meta.get("author_name", "Manager"),
                    "author_role": meta.get("author_role", "manager"),
                    "annotation_type": ann_type,
                    "created_at": meta.get("created_at"),
                    "file_path": meta.get("file_path"),
                    "symbol_name": meta.get("symbol_name"),
                    **meta,
                }
                instructions.append(instruction)

        logger.info("Retrieved %d manager instructions for project %s", len(instructions), project_id)
        return instructions

    def clear_project_annotations(
        self,
        project_id: str,
    ) -> bool:
        """Clear all annotations for a project (destructive!).

        Args:
            project_id: The project ID

        Returns:
            True if cleared successfully
        """
        store = self._get_store(project_id)
        logger.warning(f"Clearing all annotations for project {project_id}")
        return store.clear()
