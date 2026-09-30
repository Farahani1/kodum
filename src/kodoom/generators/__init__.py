"""Code-labeled Persian skill generators (plan 1.1).

Each generator module exposes ``NAME``, ``VERSION`` and
``generate(seed, pairs_per_kind) -> list[Record]``.
"""

from kodoom.generators import dates

GENERATORS = {dates.NAME: dates}
