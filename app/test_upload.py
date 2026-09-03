import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.routes.source import router
from bson import ObjectId

app = FastAPI()
app.include_router(router)

client = TestClient(app)

def test_upload():
    # 1. Create a dummy notebook in DB
    from app.database.mongodb import notebooks_collection
    import asyncio
    
    async def create_nb():
        res = await notebooks_collection.insert_one({"title": "Test Notebook"})
        return str(res.inserted_id)

    # Use existing event loop if possible, or run
    loop = asyncio.get_event_loop()
    notebook_id = loop.run_until_complete(create_nb())

    # 2. Upload a file
    with open("test_upload.pdf", "wb") as f:
        f.write(b"%PDF-1.4 dummy pdf content")

    with open("test_upload.pdf", "rb") as f:
        res = client.post(
            f"/notebooks/{notebook_id}/sources/upload",
            files={"file": ("test_upload.pdf", f, "application/pdf")}
        )
    
    print("Upload response status:", res.status_code)
    print("Upload response body:", res.json())

if __name__ == "__main__":
    test_upload()
