"""Custom generators used by the advanced demo specs.

Shows both styles supported by syntab:
  * `SkuGenerator`   — a stateful `BaseGenerator` subclass (per-run counter).
  * `local_title`    — a `@register_generator` function reading the parent row.
  * `region_phone`   — a function deriving a value from a joined parent.

Referenced from specs via import path, e.g. `demo_generators:SkuGenerator`.
The Spec file's directory is added to sys.path automatically, so this module
(next to the Spec) is importable.
"""
from syntab import BaseGenerator, register_generator


class SkuGenerator(BaseGenerator):
    """Stateful SKU generator: LIST-00001, LIST-00002, ..."""

    def __init__(self):
        self._n = 0

    def generate(self, row, rng, faker, params):
        self._n += 1
        prefix = params.get("prefix", "SKU")
        return f"{prefix}-{self._n:05d}"


@register_generator("local_title")
def local_title(row, rng, faker, params):
    # Derive the listing title's language tag from the *parent* seller's country.
    country = row["__parents__"]["seller"]["country"]
    lang = {"US": "en", "UK": "en", "DE": "de", "FR": "fr"}.get(country, "en")
    return f"[{lang}] {faker.bs()}"


@register_generator("region_phone")
def region_phone(row, rng, faker, params):
    # Phone number whose country code follows the parent seller's region.
    region = row["__parents__"]["seller"]["region"]
    cc = {"EU": "0049", "US": "001", "APAC": "0081"}.get(region, "001")
    return cc + "".join(str(rng.randint(0, 9)) for _ in range(9))
