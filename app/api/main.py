import logfire
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.notebook import router as notebook_router
from app.api.routes.source import router as source_router
from app.api.routes.user import router as user_router


app = FastAPI(
    title="NotebookLM API",
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
