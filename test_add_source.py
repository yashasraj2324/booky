import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.routes.source import router
from bson import ObjectId

app = FastAPI()
app.include_router(router)

client = TestClient(app)

def test_add_source():
    from app.database.mongodb import notebooks_collection
    
    async def create_nb():
        res = await notebooks_collection.insert_one({"title": "Test Notebook 2"})
        return str(res.inserted_id)

    loop = asyncio.get_event_loop()
    notebook_id = loop.run_until_complete(create_nb())

    res = client.post(
        f"/notebooks/{notebook_id}/sources",
        json={
            "name": "My Web Link",
            "url": "https://youtube.com/watch?v=123",
            "source_type": "youtube"
        }
    )
    print("Web response:", res.status_code, res.json())

    res2 = client.post(
        f"/notebooks/{notebook_id}/sources",
        json={
            "name": "My Text",
            "text": "This is some plain text.",
            "source_type": "text"
        }
    )
    print("Text response:", res2.status_code, res2.json())

if __name__ == "__main__":
    test_add_source()
