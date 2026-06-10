import asyncio
import os
import sys
from pathlib import Path
from pprint import pprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from backend.database import get_mongo_db, close_connections
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    sys.exit(1)

async def check_db():
    print("Checking database content...")
    db = get_mongo_db()
    count = await db.Taxpayers.count_documents({})
    print(f"Taxpayers count: {count}")
    
    if count > 0:
        cursor = db.Taxpayers.find({}, {"_id": 0, "GSTIN": 1, "Name": 1}).limit(5)
        docs = await cursor.to_list(length=5)
        print("First 5 records:")
        pprint(docs)
    else:
        print("No taxpayers found!")

    await close_connections()

if __name__ == "__main__":
    asyncio.run(check_db())
