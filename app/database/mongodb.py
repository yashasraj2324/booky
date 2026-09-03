from pymongo import AsyncMongoClient
from gridfs import AsyncGridFSBucket

from .config import settings


client = AsyncMongoClient(settings.mongodb_uri)

db = client[settings.mongodb_database]
users_collection = db["users"]
notebooks_collection = db["notebooks"]
sources_collection = db["sources"]
documents_collection = db["documents"]

fs = AsyncGridFSBucket(db)


async def connect_db() -> None:
    await client.admin.command("ping")
    print("MongoDB connected")


async def close_db() -> None:
    await client.close()
    print("MongoDB connection closed")