"""Regression tests for column-NAME-based PII detection.

Before this change the name signal could not fire at all. `_faker_provider_for`
-- the only rule in the profiler that recognises an identifier from its column
name -- was reached exclusively from inside `if is_pii:`, and the only things
that could set `is_pii` were an explicit `--pii` list and three value regexes
(email, SSN, phone) applied to `str` columns. So the name rule could never
*flag* a column; it could only pick a faker provider for a column something
else had already flagged.

The columns that fell through were exactly the ones with no distinctive value
format: `patient_name`, `home_address`, `date_of_birth`, `first_name`. No regex
matches a human name or a street address.

These tests assert the two signals are independent, that either is sufficient,
that the spec records which one fired, and that name matching is token-based
rather than substring-based (the `paid`/`squid` failure mode from
`_is_id_like`, applied to `name`).
"""
import pandas as pd
import pytest

from syntab.engine import GenerationEngine
from syntab.profiler import (
    PII_SIGNAL_EXPLICIT,
    PII_SIGNAL_NAME,
    PII_SIGNAL_VALUE,
    DatasetProfiler,
    pii_name_signal,
)


def _profile(df, **kw):
    return DatasetProfiler(df, name="t", sample=None, **kw).profile()


def _col(df, colname, **kw):
    spec = _profile(df, **kw)
    return {c.name: c for c in spec.tables[0].columns}[colname]


# ---------------------------------------------------------------------------
# the headline bug: identifiers with no distinctive value format
# ---------------------------------------------------------------------------

def test_patient_name_is_flagged():
    """The named case. Values are ordinary words; only the name identifies it."""
    df = pd.DataFrame({"patient_name": ["Alice Smith", "Bob Jones", "Carol Wu"]})
    col = _col(df, "patient_name")
    assert col.pii is True
    assert PII_SIGNAL_NAME in (col.pii_detected_by or [])
    assert col.generator == "faker.name"


def test_home_address_is_flagged():
    df = pd.DataFrame({"home_address": ["1 High St", "2 Low Rd", "3 Mid Ave"]})
    col = _col(df, "home_address")
    assert col.pii is True
    assert PII_SIGNAL_NAME in (col.pii_detected_by or [])
    assert col.generator == "faker.address"


@pytest.mark.parametrize("name,provider", [
    ("patient_name", "name"),
    ("first_name", "first_name"),
    ("FirstName", "first_name"),
    ("last_name", "last_name"),
    ("surname", "last_name"),
    ("full name", "name"),
    ("customer-name", "name"),
    ("home_address", "address"),
    ("addressLine1", "address"),
    ("street_address", "street_address"),
    ("postcode", "postcode"),
    ("postal_code", "postcode"),
    ("zip_code", "zipcode"),
    ("date_of_birth", "date_of_birth"),
    ("DOB", "date_of_birth"),
    ("birth_date", "date_of_birth"),
    ("ssn", "ssn"),
    ("social_security_number", "ssn"),
    ("passport_number", "passport_number"),
    ("driver_licence", "license_plate"),
    ("license_plate", "license_plate"),
    ("account_number", "bban"),
    ("bank_account", "bban"),
    ("iban", "iban"),
    ("credit_card_number", "credit_card_number"),
    ("medical_record_number", "bothify"),
    ("MRN", "bothify"),
    ("home_phone", "phone_number"),
    ("mobile", "phone_number"),
    ("fax_number", "phone_number"),
    ("email", "email"),
    ("contact_email_address", "email"),
    ("ip_address", "ipv4"),
    ("mac_address", "mac_address"),
    ("latitude", "latitude"),
    ("username", "user_name"),
])
def test_identifying_names_are_recognised(name, provider):
    assert pii_name_signal(name) == provider, name


# ---------------------------------------------------------------------------
# token matching, not substring matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    # a *thing's* name, qualified
    "file_name", "table_name", "column_name", "product_name", "event_name",
    "job_name", "schema_name", "service_name", "tag_name", "metric_name",
    # words that merely contain a pattern as a substring
    "filename", "nameplate", "zippy", "unaddressed", "mainstream",
    "streetcar_id",
    # unrelated
    "total", "region", "status", "amount", "quantity", "created_at",
    "", "   ",
])
def test_non_identifying_names_are_not_flagged(name):
    assert pii_name_signal(name) is None, name


def test_substring_match_would_have_produced_false_positives():
    """The property, stated directly: substring containment is not the rule.

    `"name" in "filename"` and `"zip" in "zippy"` are both true. The old
    `_faker_provider_for` used exactly that test. Tokenizing is what makes
    `patient_name` matchable without dragging `filename` along with it.
    """
    assert "name" in "filename" and pii_name_signal("filename") is None
    assert "zip" in "zippy" and pii_name_signal("zippy") is None
    assert pii_name_signal("patient_name") == "name"


# ---------------------------------------------------------------------------
# the two signals are independent
# ---------------------------------------------------------------------------

def test_name_signal_alone_is_sufficient():
    """No value regex can match these values; the name is the only evidence."""
    df = pd.DataFrame({"patient_name": ["Alice", "Bob", "Carol", "Dan"]})
    col = _col(df, "patient_name")
    assert col.pii_detected_by == [PII_SIGNAL_NAME]


def test_value_signal_alone_is_sufficient():
    """No name pattern matches `contact_field`; the values are the evidence."""
    df = pd.DataFrame({"contact_field": ["a@b.com", "c@d.com", "e@f.com"]})
    col = _col(df, "contact_field")
    assert col.pii_detected_by == [PII_SIGNAL_VALUE]
    assert col.generator == "faker.email"


def test_both_signals_are_recorded_when_both_fire():
    df = pd.DataFrame({"user_email": ["a@b.com", "c@d.com", "e@f.com"]})
    col = _col(df, "user_email")
    assert col.pii_detected_by == [PII_SIGNAL_NAME, PII_SIGNAL_VALUE]


def test_explicit_flag_is_recorded_as_its_own_signal():
    df = pd.DataFrame({"opaque": ["x", "y", "z"]})
    col = _col(df, "opaque", pii_columns=["opaque"], pii_strategy="hash")
    assert col.pii is True
    assert col.pii_detected_by == [PII_SIGNAL_EXPLICIT]
    assert col.pii_strategy == "hash"


def test_value_signal_wins_the_provider_when_the_name_disagrees():
    """`email_backup` holding phone numbers should synthesize phone numbers."""
    df = pd.DataFrame({"email_backup": ["555-123-4567", "555-987-6543",
                                        "555-222-3333"]})
    col = _col(df, "email_backup")
    assert set(col.pii_detected_by) == {PII_SIGNAL_NAME, PII_SIGNAL_VALUE}
    assert col.generator == "faker.phone_number"


# ---------------------------------------------------------------------------
# the name signal is not restricted to str columns
# ---------------------------------------------------------------------------

def test_non_string_columns_can_be_flagged_by_name():
    """The value signal only ever ran on `str` columns. The name signal does not."""
    df = pd.DataFrame({
        "date_of_birth": pd.to_datetime(
            ["1980-01-01", "1975-06-15", "1990-12-31"]),
        "phone": [15551234567, 15559876543, 15552223333],
    })
    spec = _profile(df)
    cols = {c.name: c for c in spec.tables[0].columns}
    assert cols["date_of_birth"].pii is True
    assert cols["phone"].pii is True
    assert cols["phone"].pii_detected_by == [PII_SIGNAL_NAME]


# ---------------------------------------------------------------------------
# flagging actually strips the real values
# ---------------------------------------------------------------------------

def test_name_flagged_column_leaks_nothing_into_the_spec_or_the_output():
    df = pd.DataFrame({
        "id": range(1, 41),
        "patient_name": [f"RealPatient{i}" for i in range(1, 41)],
        "home_address": [f"{i} Real Street" for i in range(1, 41)],
    })
    spec = _profile(df)
    cols = {c.name: c for c in spec.tables[0].columns}
    for name in ("patient_name", "home_address"):
        assert cols[name].profile is None
        assert cols[name].params == {}

    out = GenerationEngine(spec).run().to_frames()["t"]
    assert not out["patient_name"].astype(str).str.contains(
        "RealPatient", regex=False).any()
    assert not out["home_address"].astype(str).str.contains(
        "Real Street", regex=False).any()


# ---------------------------------------------------------------------------
# structural keys are reported, not silently anonymized
# ---------------------------------------------------------------------------

def test_identifying_key_columns_are_surfaced_rather_than_anonymized():
    """`patient_id` is a key. Anonymizing it would break PK/FK integrity.

    So it is not auto-flagged -- but it is not ignored either: it lands on
    `identifying_key_candidates` for the disclosure summary to put in front of
    a human.
    """
    df = pd.DataFrame({"patient_id": [1, 2, 3, 4], "score": [1.0, 2.0, 3.0, 4.0]})
    prof = DatasetProfiler(df, name="t", sample=None)
    spec = prof.profile()
    col = {c.name: c for c in spec.tables[0].columns}["patient_id"]
    assert col.pii is False
    assert spec.tables[0].primary_key == "patient_id"
    assert prof.identifying_key_candidates == ["patient_id"]


def test_foreign_keys_survive_name_based_detection():
    """A `*_id` column must keep its type and values or FK inference dies."""
    users = pd.DataFrame({"id": [1, 2, 3], "patient_name": ["a", "b", "c"]})
    visits = pd.DataFrame({"id": [10, 11], "users_id": [1, 2]})
    spec = DatasetProfiler.profile_set(
        {"users": users, "visits": visits}, sample=None)
    tables = {t.name: t for t in spec.tables}
    rels = tables["visits"].relationships
    assert len(rels) == 1 and rels[0].to == "users.id"
    # ...and the person's name in the same table still got flagged
    assert {c.name: c for c in tables["users"].columns}["patient_name"].pii
