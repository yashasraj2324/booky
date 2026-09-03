import logfire
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.notebook import router as notebook_router
from app.api.routes.source import router as source_router
from app.api.routes.user import router as user_router
from app.api.routes.chat import router as chat_router
from app.database.mongodb import connect_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and close the MongoDB connection on startup/shutdown."""
    await connect_db()
    yield
    await close_db()


app = FastAPI(
    title="NotebookLM API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Logfire
logfire.configure(console=False)

# Instrument FastAPI
logfire.instrument_fastapi(app)

# Routes
app.include_router(user_router)
app.include_router(
    notebook_router,
    prefix="/notebooks",
    tags=["notebooks"],
)

app.include_router(source_router)

app.include_router(chat_router)
