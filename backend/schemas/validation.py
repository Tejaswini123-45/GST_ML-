"""
Pydantic schemas for input validation.
"""

import re
from pydantic import BaseModel, Field, field_validator


GSTIN_PATTERN = re.compile(r'^[a-zA-Z0-9]{14,15}$')
INVOICE_PATTERN = re.compile(r'^[A-Za-z0-9\-_]{1,50}$')


class GSTINQuery(BaseModel):
    gstin: str

    @field_validator('gstin')
    @classmethod
    def validate_gstin(cls, v):
        v = v.strip()
        if not GSTIN_PATTERN.match(v):
            raise ValueError(
                "Invalid GSTIN format. Expected 14-15 alphanumeric characters."
            )
        return v


class InvoiceQuery(BaseModel):
    invoice_id: str

    @field_validator('invoice_id')
    @classmethod
    def validate_invoice_id(cls, v):
        v = v.strip()
        if not INVOICE_PATTERN.match(v):
            raise ValueError(
                "Invalid Invoice ID. Use only letters, digits, hyphens, underscores (max 50 chars)."
            )
        return v


def validate_gstin_param(gstin: str) -> str:
    """Validate a GSTIN path parameter. Raises ValueError if invalid."""
    gstin = gstin.strip()
    if not GSTIN_PATTERN.match(gstin):
        raise ValueError(f"Invalid GSTIN: '{gstin}'")
    return gstin


def validate_invoice_param(invoice_id: str) -> str:
    """Validate an invoice_id path parameter. Raises ValueError if invalid."""
    invoice_id = invoice_id.strip()
    if not INVOICE_PATTERN.match(invoice_id):
        raise ValueError(f"Invalid Invoice ID: '{invoice_id}'")
    return invoice_id
