from database.mongodb import connect_db, close_db

import asyncio


async def main():
    await connect_db()
    print("Hello from app! MongoDB connected")
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
