"""Mock banking API, based on the case README's mock_payment(vendor, amount).

Extended (per the payment requirements) so the 'bank' also receives the currency, the invoice
number and our payment reference. Prints to stderr so `--json` output on stdout stays valid.
"""

import sys


def mock_payment(vendor: str, amount: float, currency: str | None, invoice_number: str, reference: str) -> dict:
    print(f"Paid {amount:,.2f} {currency or '(currency not stated)'} to {vendor} "
          f"for {invoice_number} (ref {reference})", file=sys.stderr)
    return {"status": "success", "reference": reference}
