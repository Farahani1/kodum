"""Which parts of a typed-decisions case are translated and which are kept (plan 1.1).

The table below was decided from ``kodoom fields`` over all 1,200 train and test
cases: free text is translated, and everything that code, a join or a reader may
compare byte for byte is kept. A path is a state key path with list positions
collapsed to ``[]``, as ``kodoom fields`` prints them. A text field that is not in
the table is an error, not a guess, so a new dataset revision cannot slip
untranslated or mistranslated text through.

Beyond the state: question text and option descriptions are always translated
(they are model input); option ids, JSON keys and gold are never touched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

TRANSLATE = "translate"
KEEP = "keep"
# Register rules (plan 1.1): customer messages colloquial, the rest formal.
COLLOQUIAL = "colloquial"
FORMAL = "formal"


class RuleError(ValueError):
    """A case has a text field the rules do not cover, or a broken structure."""


@dataclass(frozen=True)
class Rule:
    action: str
    register: str = FORMAL


_T = Rule(TRANSLATE)
_K = Rule(KEEP)

RULES: dict[str, dict[str, Rule]] = {
    "agent_trace_observability": {
        "task": _T,
        "constraints[]": _T,
        "agent.model": _K,
        "agent.autonomy": _K,
    },
    "customer_service": {
        "thread[].text": Rule(TRANSLATE, COLLOQUIAL),
        "thread[].role": _K,
        "orders[].id": _K,
        "orders[].date": _K,
        "orders[].status": _K,
        "account.tier": _K,
    },
    "invoice_processing": {
        "purchase_order.freight_terms": _T,
        "delivery.condition": _T,
        "vendor_history.prior_invoice_ids[]": _K,
        "invoice.id": _K,
        "invoice.vendor": _K,
        "invoice.currency": _K,
        "invoice.lines[].sku": _K,
        "purchase_order.id": _K,
        "delivery.date": _K,
        "payment.terms": _K,
        "payment.status": _K,
    },
    "security_incidents": {
        "alert.evidence": _T,
        "alert.description": _T,
        "alert.rule": _K,
        "principal.name": _K,
        "principal.type": _K,
        "principal.privileges[]": _K,
        "context.asset_criticality": _K,
    },
}


@dataclass(frozen=True)
class Segment:
    """One text leaf of a state: where it is, what it says, and what to do with it."""

    location: tuple[str | int, ...]
    path: str
    text: str
    rule: Rule


def walk(node: Any, location: tuple[str | int, ...] = (), path: str = "") -> Iterator[tuple]:
    """Text leaves as (location, path, text); non-text leaves are always kept."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, (*location, key), f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from walk(value, (*location, i), f"{path}[]")
    elif isinstance(node, str):
        yield location, path, node


def segments(workflow: str, state: str) -> list[Segment]:
    """Every text leaf of a state with its rule; unknown workflows and paths raise."""
    rules = RULES.get(workflow)
    if rules is None:
        raise RuleError(f"no translation rules for workflow {workflow!r}")
    try:
        tree = json.loads(state)
    except json.JSONDecodeError as e:
        raise RuleError(f"state is not valid JSON: {e}") from e
    found = []
    for location, path, text in walk(tree):
        if path not in rules:
            raise RuleError(f"{workflow}: text field {path!r} has no rule; add it to RULES")
        found.append(Segment(location, path, text, rules[path]))
    return found


def apply(state: str, translated: Mapping[tuple[str | int, ...], str]) -> str:
    """The state with the given leaves replaced; everything else is unchanged.

    Keys keep their order and the JSON stays valid; non-ASCII text is written as is.
    """
    tree = json.loads(state)
    for location, text in translated.items():
        node = tree
        for step in location[:-1]:
            node = node[step]
        if not isinstance(node[location[-1]], str):
            raise RuleError(f"{location!r} is not a text field")
        node[location[-1]] = text
    return json.dumps(tree, ensure_ascii=False)
