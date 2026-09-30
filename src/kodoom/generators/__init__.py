"""Code-labeled Persian skill generators (plan 1.1).

Each generator module exposes ``NAME``, ``VERSION`` and
``generate(seed, pairs_per_kind) -> list[Record]``.
"""

from kodoom.generators import currency, dates, digits, formats, hours

GENERATORS = {module.NAME: module for module in (dates, digits, currency, hours, formats)}
