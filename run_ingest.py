import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from backend.ingestion import ingest_all_files
    from backend.database import close_connections, get_mongo_db
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    sys.exit(1)

async def main():
    print("Starting data ingestion...")
    try:
        report = await ingest_all_files()
        print("Ingestion Report:")
        for filename, status in report.items():
            print(f"  {filename}: {status}")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await close_connections()

if __name__ == "__main__":
    asyncio.run(main())
