import asyncio
from pymongo import AsyncMongoClient

async def fix():
    from app.database.config import settings
    client = AsyncMongoClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]
    await db['notebooks'].delete_many({'created_at': {'$exists': False}})
    print("Deleted malformed notebooks")
    await client.close()

if __name__ == "__main__":
    asyncio.run(fix())
