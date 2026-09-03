import asyncio
from app.models.users import UserRegister
from app.api.routes.user import register
import app.database.mongodb as mdb

async def main():
    await mdb.connect_db()
    u = UserRegister(first_name="f", last_name="l", email="user4@example.com", phone_number="1234567890", password="password")
    res = await register(u)
    print(res)

asyncio.run(main())
