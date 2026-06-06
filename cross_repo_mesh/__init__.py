"""Vendored cross-repo ripple mesh (sibling: jarvis-prime).

ripple_contract.py is a BYTE-IDENTICAL vendored copy of JARVIS's portable
verification contract (backend/core/ouroboros/cross_repo_mesh/ripple_contract.py).
Independent verification by design — this repo trusts a JARVIS ripple ONLY
after verifying its HMAC/nonce/TTL/origin locally, and NEVER executes anything
JARVIS sends ("predictions, not requests").
"""
