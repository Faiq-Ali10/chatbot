"""
ServiceAI Chatbot Backend
=========================
A highly performant, asynchronous FastAPI application powering the ServiceAI
informational chatbot. Integrates Groq via LangChain and enforces
strict request/response validation through Pydantic models.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Environment & Logging Setup
# ---------------------------------------------------------------------------

load_dotenv()  # Load GOOGLE_API_KEY (and any other secrets) from .env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("serviceai.chatbot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_PATH: Path = Path(__file__).parent / "system_prompt.txt"
HISTORY_MESSAGE_LIMIT: int = 6   # Exact number of historical messages expected
GROQ_MODEL: str = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """Represents a single message in the conversation history."""

    role: Literal["user", "assistant"] = Field(
        ...,
        description="The author of the message. Must be either 'user' or 'assistant'.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The text content of the message. Cannot be empty.",
    )

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content must not be blank or whitespace-only.")
        return v.strip()


class ChatRequest(BaseModel):
    """
    Incoming payload sent by the frontend.

    The frontend must supply exactly the last 6 messages of the conversation
    history (alternating user / assistant turns) along with the new user query.
    """

    history: list[ChatMessage] = Field(
        ...,
        min_length=0,
        max_length=HISTORY_MESSAGE_LIMIT,
        description=(
            f"The last {HISTORY_MESSAGE_LIMIT} messages of the conversation history. "
            "Ordered from oldest to newest. Maximum of 6 messages allowed."
        ),
    )
    query: str = Field(
        ...,
        min_length=1,
        description="The new user query to send to the assistant.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be blank or whitespace-only.")
        return v.strip()


class ChatResponse(BaseModel):
    """Outgoing payload returned to the frontend."""

    response: str = Field(
        ...,
        description="The AI-generated response from the assistant.",
    )

# ---------------------------------------------------------------------------
# System Prompt Loader  (cached - file is read once per process lifetime)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """
    Read the system prompt from disk.

    Cached via @lru_cache so the file is only read once regardless of
    how many concurrent requests arrive.

    Raises:
        RuntimeError: If the system_prompt.txt file is missing or empty.
    """
    if not SYSTEM_PROMPT_PATH.exists():
        raise RuntimeError(
            f"System prompt file not found at: {SYSTEM_PROMPT_PATH}. "
            "Ensure 'system_prompt.txt' exists in the same directory as main.py."
        )

    content = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    if not content:
        raise RuntimeError(
            f"System prompt file is empty: {SYSTEM_PROMPT_PATH}. "
            "Please populate 'system_prompt.txt' with a valid system prompt."
        )

    logger.info("System prompt loaded successfully (%d characters).", len(content))
    return content


# ---------------------------------------------------------------------------
# LLM Factory  (cached - one model instance shared across all requests)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Instantiate and return the LangChain Groq model.

    Cached so only one instance is created for the entire application lifetime,
    avoiding unnecessary re-initialisation overhead on each request.

    Raises:
        RuntimeError: If GROQ_API_KEY is not set in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set. "
            "Add it to your .env file or export it in your shell."
        )

    llm = ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=api_key,
        temperature=0,
    )
    logger.info("LLM initialised: model=%s", GROQ_MODEL)
    return llm


# ---------------------------------------------------------------------------
# Message Assembly Helper
# ---------------------------------------------------------------------------


def build_message_chain(
    system_prompt: str,
    history: list[ChatMessage],
    query: str,
) -> list:
    """
    Assemble the full LangChain message list to be sent to the model.

    Order:
        1. SystemMessage  - the ServiceAI persona and instructions
        2. Historical messages - up to 6 prior turns (HumanMessage / AIMessage)
        3. HumanMessage   - the current user query

    Args:
        system_prompt: Raw system prompt string loaded from file.
        history:       Ordered list of prior ChatMessage objects.
        query:         The new user question/input.

    Returns:
        A list of LangChain message objects ready for model invocation.
    """
    messages = [SystemMessage(content=system_prompt)]

    role_map = {
        "user": HumanMessage,
        "assistant": AIMessage,
    }

    for msg in history:
        lc_message_cls = role_map[msg.role]
        messages.append(lc_message_cls(content=msg.content))

    # Append the live user query as the final human turn
    messages.append(HumanMessage(content=query))

    return messages


# ---------------------------------------------------------------------------
# Application Lifespan  (startup / shutdown hooks)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up shared resources on startup; clean up on shutdown."""
    logger.info("=== ServiceAI Chatbot API - Starting Up ===")
    try:
        # Eagerly validate that the system prompt and LLM are available
        load_system_prompt()
        get_llm()
        logger.info("All startup checks passed. API is ready to serve requests.")
    except RuntimeError as exc:
        logger.critical("Startup failed: %s", exc)
        raise  # Abort startup if critical resources are missing

    yield  # Application runs here

    logger.info("=== ServiceAI Chatbot API - Shutting Down ===")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="ServiceAI Chatbot API",
    description=(
        "Asynchronous AI chatbot backend powering the ServiceAI informational assistant. "
        "Built with FastAPI and LangChain, backed by Groq."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS - adjust origins for your production frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
async def root() -> dict:
    """Basic health-check endpoint."""
    return {"status": "ok", "service": "ServiceAI Chatbot API"}


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Detailed health-check that confirms core dependencies are operational.
    Returns HTTP 503 if any critical resource is unavailable.
    """
    try:
        load_system_prompt()
        get_llm()
        return {"status": "healthy", "model": GROQ_MODEL}
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the ServiceAI assistant",
    tags=["Chat"],
    responses={
        200: {"description": "AI response returned successfully."},
        422: {"description": "Validation error - check your request payload."},
        500: {"description": "Internal server error during model invocation."},
        503: {"description": "Service unavailable - model or config not ready."},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    **Primary chat endpoint.**

    Accepts the last <=6 historical messages and a new user query.
    Assembles the full message chain (system prompt + history + query),
    invokes the Gemini model asynchronously, and returns the AI response.

    ### Request Body
    ```json
    {
      "history": [
        {"role": "user",      "content": "What services do you offer?"},
        {"role": "assistant", "content": "We offer custom software development..."},
      ],
      "query": "Do you work with startups?"
    }
    ```

    ### Response Body
    ```json
    {
      "response": "Absolutely! ServiceAI loves partnering with startups..."
    }
    ```
    """
    logger.info(
        "Incoming /chat request | history_len=%d | query_len=%d",
        len(request.history),
        len(request.query),
    )

    # --- Load cached resources ---
    try:
        system_prompt = load_system_prompt()
        llm = get_llm()
    except RuntimeError as exc:
        logger.error("Resource loading failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # --- Assemble message chain ---
    messages = build_message_chain(
        system_prompt=system_prompt,
        history=request.history,
        query=request.query,
    )

    logger.debug("Message chain assembled: %d messages total.", len(messages))

    # --- Invoke model asynchronously ---
    try:
        ai_message = await llm.ainvoke(messages)
    except Exception as exc:
        logger.exception("Model invocation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Our AI assistant is currently undergoing maintenance. Please use the contact form and our team will contact you shortly.",
        ) from exc

    response_text: str = ai_message.content
    logger.info("Response generated successfully (%d characters).", len(response_text))

    return ChatResponse(response=response_text)
