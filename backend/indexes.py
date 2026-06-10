"""
MongoDB index management.

Call ensure_indexes() once at application startup.
All indexes are created with background=True so they don't block startup.
"""

import asyncio
from backend.database import get_mongo_db
from backend.utils.logger import get_logger

logger = get_logger(__name__)


async def ensure_indexes() -> None:
    """Create all required MongoDB indexes (idempotent — safe to re-run)."""
    db = get_mongo_db()

    index_spec: list[tuple[str, list[tuple[str, int]]]] = [
        # Taxpayers — primary lookup by GSTIN
        ("Taxpayers", [("GSTIN", 1)]),

        # Invoices — lookups by seller, buyer, and invoice ID
        ("Invoices", [("Invoice_ID", 1)]),
        ("Invoices", [("Seller_GSTIN", 1)]),
        ("Invoices", [("Buyer_GSTIN", 1)]),
        ("Invoices", [("Seller_GSTIN", 1), ("Buyer_GSTIN", 1)]),

        # GSTR1 — lookup by invoice and seller
        ("GSTR1", [("Invoice_ID", 1)]),
        ("GSTR1", [("Seller_GSTIN", 1)]),

        # GSTR2B — lookup by buyer and invoice
        ("GSTR2B", [("Buyer_GSTIN", 1)]),
        ("GSTR2B", [("Invoice_ID", 1)]),
        ("GSTR2B", [("Buyer_GSTIN", 1), ("ITC_Eligible", 1)]),

        # GSTR3B — lookup by seller
        ("GSTR3B", [("Seller_GSTIN", 1)]),
        ("GSTR3B", [("Seller_GSTIN", 1), ("Payment_Confirmed", 1)]),

        # EWayBill — lookup by invoice
        ("EWayBill", [("Invoice_ID", 1)]),
        ("EWayBill", [("EWayBill_No", 1)]),

        # Purchase_Register — lookup by buyer and invoice
        ("Purchase_Register", [("Buyer_GSTIN", 1)]),
        ("Purchase_Register", [("Invoice_ID", 1)]),

        # Ingestion log — lookup by file name
        ("_ingestion_log", [("file", 1)]),
    ]

    created = 0
    errors = 0

    for collection_name, keys in index_spec:
        try:
            col = db[collection_name]
            # build pymongo-style key list for create_index
            await col.create_index(keys, background=True, sparse=True)
            created += 1
        except Exception as exc:
            errors += 1
            logger.warning("Index creation skipped (%s %s): %s", collection_name, keys, exc)

    logger.info(
        "MongoDB indexes: %d created/confirmed, %d errors.", created, errors
    )
