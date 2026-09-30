import pytest

from kodoom.translate import glossary
from kodoom.translate.glossary import GlossaryError, parse


@pytest.fixture(scope="module")
def g():
    return glossary.load()


def test_the_packaged_glossary_loads_and_has_every_workflow(g):
    assert {"refund", "invoice"} <= set(g.common)
    for workflow in (
        "customer_service",
        "agent_trace_observability",
        "invoice_processing",
        "security_incidents",
    ):
        assert g.terms(workflow), workflow
    assert "TLS" in g.keep


def test_agent_means_two_things(g):
    assert g.terms("customer_service")["agent"] != g.terms("agent_trace_observability")["agent"]
    assert "agent" not in g.terms("security_incidents")


def test_relevant_matches_whole_words_and_plurals_only(g):
    found = g.relevant("customer_service", "Refunds for the Invoice, not for reagents.")
    assert set(found) == {"refund", "invoice"}
    assert g.relevant("customer_service", "The agents wait.") == {
        "agent": g.terms("customer_service")["agent"]
    }


def test_check_passes_when_the_persian_term_is_used(g):
    ok = "بازپرداخت فاکتور شما انجام شد"
    assert g.check("customer_service", "Your refund of the invoice is done", ok) == []
    assert g.check("customer_service", "Refunds", "بازپرداخت\u200cها") == []  # inflection
    assert g.check("customer_service", "Your vendor", "تأمین کننده") == []  # ZWNJ ignored


def test_check_flags_a_missing_term_and_a_translated_keep_term(g):
    findings = g.check("customer_service", "Your refund", "پول شما برگشت", "state.x")
    assert [(f.check, f.where) for f in findings] == [("glossary", "state.x")]
    assert "refund" in findings[0].message
    kept = g.check("agent_trace_observability", "Rotate the TLS certificate", "گواهی را عوض کنید")
    assert "TLS" in kept[0].message
    assert g.check("agent_trace_observability", "Rotate the TLS certificate", "گواهی TLS") == []
    assert g.check("agent_trace_observability", "the TLSv1 stack", "پشته") == []  # not whole


def test_bad_glossaries_are_refused():
    with pytest.raises(GlossaryError, match="invalid TOML"):
        parse("[common\n")
    with pytest.raises(GlossaryError, match="lower case"):
        parse('[common]\n"Refund" = "بازپرداخت"\n')
    with pytest.raises(GlossaryError, match="Persian text"):
        parse('[common]\n"refund" = ""\n')
    with pytest.raises(GlossaryError, match="terms must be"):
        parse("[keep]\nterms = [1]\n")
