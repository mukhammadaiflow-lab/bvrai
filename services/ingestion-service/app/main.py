"""
Ingestion & Persona Generator Service - Main FastAPI Application.

Handles:
- Website scraping and content extraction
- Text chunking and embedding
- Persona/system prompt generation
- Few-shot dialogue generation
"""
import asyncio
import base64
import hashlib
import json
import re
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import uuid4

import httpx
import numpy as np
import structlog
from bs4 import BeautifulSoup
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()


# ============================================================================
# Configuration
# ============================================================================

class Settings(BaseSettings):
    """Application settings."""
    port: int = 3004
    host: str = "0.0.0.0"
    environment: str = "development"
    vector_db_path: str = "./data/vectors.db"
    tenant_config_path: str = "./data/tenants"
    max_urls_per_request: int = 10
    max_chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"


settings = Settings()


# ============================================================================
# Models
# ============================================================================

class UploadFile(BaseModel):
    """Uploaded file content."""
    filename: str
    content: str  # base64 encoded


class IngestRequest(BaseModel):
    """Ingestion request."""
    tenant_id: str = Field(..., min_length=1, max_length=128)
    urls: list[str] = Field(default_factory=list, max_length=10)
    uploads: list[UploadFile] = Field(default_factory=list, max_length=10)


class TaskInfo(BaseModel):
    """Task information."""
    task_id: str
    status: str = "pending"


class IngestResponse(BaseModel):
    """Ingestion response."""
    status: str = "accepted"
    tasks: list[TaskInfo]


class PersonaConfig(BaseModel):
    """Generated persona configuration."""
    tenant_id: str
    system_prompt: str
    few_shots: list[dict[str, str]]
    facts: list[str]
    generated_at: str


# ============================================================================
# Services
# ============================================================================

@dataclass
class TextChunk:
    """A chunk of text from a document."""
    id: str
    content: str
    source: str
    metadata: dict[str, Any]


class WebScraper:
    """
    Web scraper for extracting content from URLs.

    Uses httpx for async HTTP requests and BeautifulSoup for HTML parsing.
    """

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "BVRAI-Ingestion/1.0"},
        )

    async def scrape_url(self, url: str) -> tuple[str, dict[str, Any]]:
        """
        Scrape content from a URL.

        Returns:
            Tuple of (extracted text, metadata)
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Extract title
            title = soup.title.string if soup.title else ""

            # Extract main content
            # Try common content containers first
            content_element = (
                soup.find("main")
                or soup.find("article")
                or soup.find(class_=re.compile(r"content|main|article", re.I))
                or soup.body
            )

            text = ""
            if content_element:
                text = content_element.get_text(separator="\n", strip=True)

            # Clean up text
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)

            metadata = {
                "url": url,
                "title": title,
                "scraped_at": datetime.utcnow().isoformat(),
            }

            logger.info("scraped_url", url=url, text_length=len(text))
            return text, metadata

        except Exception as e:
            logger.error("scrape_error", url=url, error=str(e))
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class TextChunker:
    """
    Text chunker for splitting documents into smaller pieces.

    Uses a simple character-based chunking with overlap.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: str, metadata: dict[str, Any]) -> list[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            source: Source identifier (URL or filename)
            metadata: Additional metadata

        Returns:
            List of TextChunk objects
        """
        chunks = []

        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, source, metadata, len(chunks)))
                current_chunk = para + "\n\n"

        # Add remaining content
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, source, metadata, len(chunks)))

        logger.debug("chunked_text", source=source, num_chunks=len(chunks))
        return chunks

    def _create_chunk(
        self, content: str, source: str, metadata: dict[str, Any], index: int
    ) -> TextChunk:
        """Create a TextChunk with a unique ID."""
        chunk_id = hashlib.sha256(f"{source}:{index}:{content[:50]}".encode()).hexdigest()[:16]
        return TextChunk(
            id=f"chunk-{chunk_id}",
            content=content.strip(),
            source=source,
            metadata={**metadata, "chunk_index": index},
        )


class VectorStore:
    """
    Local SQLite-based vector store.

    Same implementation as Dialog Manager for consistency.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _generate_embedding(self, text: str, dim: int = 128) -> list[float]:
        """Generate simple hash-based embedding for testing."""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = [hash_bytes[i % len(hash_bytes)] / 255.0 for i in range(dim)]
        norm = np.linalg.norm(embedding)
        return [v / norm for v in embedding] if norm > 0 else embedding

    def upsert_chunks(self, tenant_id: str, chunks: list[TextChunk]) -> int:
        """Store chunks in the vector database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        count = 0
        for chunk in chunks:
            embedding = self._generate_embedding(chunk.content)
            metadata = {**chunk.metadata, "tenant_id": tenant_id}

            cursor.execute(
                """
                INSERT OR REPLACE INTO documents (id, content, metadata, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (chunk.id, chunk.content, json.dumps(metadata), json.dumps(embedding)),
            )
            count += 1

        conn.commit()
        conn.close()
        return count


class PersonaGenerator:
    """
    Generates tenant personas and few-shot dialogues.

    Uses a mock LLM for generation (rule-based for testing).
    TODO: Integrate real LLM for production.
    """

    def __init__(self) -> None:
        self.templates = {
            "default": {
                "system_prompt": """You are a helpful AI assistant for {business_name}.
Your role is to assist customers with inquiries, provide information, and help with common tasks.

Key Information:
{facts}

Be friendly, professional, and helpful. If you don't know something, say so politely.""",
                "few_shots": [
                    {"user": "Hello!", "assistant": "Hello! Welcome to {business_name}. How can I help you today?"},
                    {"user": "What services do you offer?", "assistant": "We offer a variety of services. Based on what I know, {services_summary}. Is there something specific you'd like to know more about?"},
                    {"user": "What are your hours?", "assistant": "Our business hours are {hours}. Is there anything else I can help you with?"},
                ],
            }
        }

    def generate_persona(self, tenant_id: str, facts: list[str]) -> PersonaConfig:
        """
        Generate a persona configuration from extracted facts.

        Args:
            tenant_id: Tenant identifier
            facts: List of facts extracted from content

        Returns:
            PersonaConfig with system prompt and few-shots
        """
        # Extract business name from facts or use default
        business_name = "our business"
        services_summary = "various services"
        hours = "standard business hours"

        for fact in facts:
            fact_lower = fact.lower()
            if "hour" in fact_lower or "open" in fact_lower:
                hours = fact
            if "service" in fact_lower or "offer" in fact_lower:
                services_summary = fact

        # Build system prompt
        facts_text = "\n".join(f"- {fact}" for fact in facts[:20])
        template = self.templates["default"]

        system_prompt = template["system_prompt"].format(
            business_name=business_name,
            facts=facts_text,
        )

        # Generate few-shots
        few_shots = []
        for shot in template["few_shots"]:
            few_shots.append({
                "user": shot["user"],
                "assistant": shot["assistant"].format(
                    business_name=business_name,
                    services_summary=services_summary,
                    hours=hours,
                ),
            })

        # Add fact-based few-shots
        for i, fact in enumerate(facts[:10]):
            if len(fact) > 50:
                few_shots.append({
                    "user": f"Tell me about {fact[:30]}...",
                    "assistant": fact,
                })

        return PersonaConfig(
            tenant_id=tenant_id,
            system_prompt=system_prompt,
            few_shots=few_shots[:20],  # Limit to 20 examples
            facts=facts[:50],
            generated_at=datetime.utcnow().isoformat(),
        )

    def save_persona(self, config: PersonaConfig, base_path: str) -> str:
        """Save persona configuration to file."""
        path = Path(base_path)
        path.mkdir(parents=True, exist_ok=True)

        filepath = path / f"{config.tenant_id}.json"
        with open(filepath, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

        return str(filepath)


# ============================================================================
# Background Tasks
# ============================================================================

# Task status storage
task_status: dict[str, dict[str, Any]] = {}

async def process_ingestion(
    task_id: str,
    tenant_id: str,
    urls: list[str],
    uploads: list[UploadFile],
) -> None:
    """Background task for processing ingestion."""
    task_status[task_id] = {"status": "processing", "progress": 0}

    scraper = WebScraper()
    chunker = TextChunker(settings.max_chunk_size, settings.chunk_overlap)
    vector_store = VectorStore(settings.vector_db_path)
    persona_gen = PersonaGenerator()

    all_chunks: list[TextChunk] = []
    all_facts: list[str] = []

    try:
        # Process URLs
        total_items = len(urls) + len(uploads)
        processed = 0

        for url in urls:
            try:
                text, metadata = await scraper.scrape_url(url)
                chunks = chunker.chunk_text(text, url, metadata)
                all_chunks.extend(chunks)

                # Extract facts (first sentence of each chunk)
                for chunk in chunks:
                    first_sentence = chunk.content.split(".")[0] + "."
                    if len(first_sentence) > 20:
                        all_facts.append(first_sentence[:200])

                processed += 1
                task_status[task_id]["progress"] = int(processed / total_items * 100)

            except Exception as e:
                logger.error("url_processing_error", url=url, error=str(e))

        # Process uploads
        for upload in uploads:
            try:
                content = base64.b64decode(upload.content).decode("utf-8")
                metadata = {"filename": upload.filename, "type": "upload"}
                chunks = chunker.chunk_text(content, upload.filename, metadata)
                all_chunks.extend(chunks)

                for chunk in chunks:
                    first_sentence = chunk.content.split(".")[0] + "."
                    if len(first_sentence) > 20:
                        all_facts.append(first_sentence[:200])

                processed += 1
                task_status[task_id]["progress"] = int(processed / total_items * 100)

            except Exception as e:
                logger.error("upload_processing_error", filename=upload.filename, error=str(e))

        # Store in vector DB
        stored_count = vector_store.upsert_chunks(tenant_id, all_chunks)

        # Generate persona
        persona = persona_gen.generate_persona(tenant_id, all_facts)
        persona_path = persona_gen.save_persona(persona, settings.tenant_config_path)

        task_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "chunks_stored": stored_count,
            "facts_extracted": len(all_facts),
            "persona_path": persona_path,
        }

        logger.info(
            "ingestion_completed",
            task_id=task_id,
            tenant_id=tenant_id,
            chunks=stored_count,
            facts=len(all_facts),
        )

    except Exception as e:
        task_status[task_id] = {"status": "failed", "error": str(e)}
        logger.error("ingestion_failed", task_id=task_id, error=str(e))

    finally:
        await scraper.close()


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan."""
    logger.info("starting_ingestion_service", port=settings.port)
    yield
    logger.info("shutting_down_ingestion_service")


app = FastAPI(
    title="Ingestion & Persona Generator Service",
    description="Content ingestion and persona generation for Builder Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ingestion-service",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_content(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Ingest content from URLs and uploads.

    Processes content in the background and returns task IDs.
    """
    if not request.urls and not request.uploads:
        raise HTTPException(status_code=400, detail="No URLs or uploads provided")

    task_id = str(uuid4())

    background_tasks.add_task(
        process_ingestion,
        task_id,
        request.tenant_id,
        request.urls,
        request.uploads,
    )

    return IngestResponse(
        status="accepted",
        tasks=[TaskInfo(task_id=task_id, status="pending")],
    )


@app.get("/ingest/status/{task_id}")
async def get_task_status(task_id: str) -> dict[str, Any]:
    """Get ingestion task status."""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"task_id": task_id, **task_status[task_id]}


@app.get("/tenants/{tenant_id}/persona")
async def get_tenant_persona(tenant_id: str) -> PersonaConfig:
    """Get generated persona for a tenant."""
    filepath = Path(settings.tenant_config_path) / f"{tenant_id}.json"

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Persona not found")

    with open(filepath) as f:
        data = json.load(f)

    return PersonaConfig(**data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
