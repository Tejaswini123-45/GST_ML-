import pytest
from pydantic import ValidationError

from backend.schemas.validation import (
    GSTINQuery,
    InvoiceQuery,
    validate_gstin_param,
    validate_invoice_param,
)


def test_gstin_is_trimmed():
    assert validate_gstin_param("  ABCDE123456789  ") == "ABCDE123456789"


@pytest.mark.parametrize("value", ["", "ABC 123456789", "GSTIN_TOO_LONG_123456"])
def test_invalid_gstin_is_rejected(value):
    with pytest.raises(ValueError):
        validate_gstin_param(value)


def test_invoice_id_is_trimmed():
    assert validate_invoice_param("  INV-15_2  ") == "INV-15_2"


@pytest.mark.parametrize("value", ["", "INV 15", "INV#15", "x" * 51])
def test_invalid_invoice_id_is_rejected(value):
    with pytest.raises(ValueError):
        validate_invoice_param(value)


def test_pydantic_models_reject_invalid_values():
    with pytest.raises(ValidationError):
        GSTINQuery(gstin="bad value")
    with pytest.raises(ValidationError):
        InvoiceQuery(invoice_id="bad value")
