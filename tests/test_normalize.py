import pytest

from kodoom.normalize import ZWNJ, clean_orthography, normalize, to_latin_digits

Z = ZWNJ


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Arabic letters to Persian
        ("علي كتاب", "علی کتاب"),
        ("مصطفى", "مصطفی"),
        # Digits: Persian and Arabic-Indic to Latin, separators between digits
        ("۱۲۳۴", "1234"),
        ("٤٥٦", "456"),
        ("۱۲٫۵", "12.5"),
        ("۱٬۰۰۰٬۰۰۰ ریال", "1,000,000 ریال"),
        ("۲۰٪", "20%"),
        # Diacritics and tatweel
        ("حتماً", "حتما"),
        ("كـــتاب", "کتاب"),
        # Verb prefix می / نمی
        ("می خواهم", f"می{Z}خواهم"),
        ("نمی دانم", f"نمی{Z}دانم"),
        ("او می  رود", f"او می{Z}رود"),
        # Plural suffixes
        ("کتاب ها", f"کتاب{Z}ها"),
        ("فاکتور های من", f"فاکتور{Z}های من"),
        ("درخواست هایشان", f"درخواست{Z}هایشان"),
        # Spaces and stray ZWNJ
        ("سلام   دنیا", "سلام دنیا"),
        ("سلام\u00a0دنیا", "سلام دنیا"),
        (f"سلام{Z} دنیا", "سلام دنیا"),
        (f"{Z}سلام{Z}", "سلام"),
        (f"می{Z}{Z}خواهم", f"می{Z}خواهم"),
        ("  خط اول  \r\n  خط دوم  ", "خط اول\nخط دوم"),
        # Invisible marks
        ("\ufeffسلام\u200f", "سلام"),
        # Presentation forms (e.g. text copied from old PDFs)
        ("\ufedf\ufe8e", "لا"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Refund invoice INV-2291 to billing@example.com",
        "GET https://api.example.com/v1/tickets?id=42 -> 500",
        "ERROR 2026-09-30T10:15:00Z worker-3: timeout",
    ],
)
def test_english_and_keep_fields_pass_through(text):
    assert normalize(text) == text


def test_does_not_join_words_that_merely_start_with_mi():
    # "میز" (table) and "میوه" (fruit) are words, not the verb prefix.
    assert normalize("میز بزرگ") == "میز بزرگ"
    assert normalize("میوه ها") == f"میوه{Z}ها"
    # "ها" inside a word is not a suffix.
    assert normalize("هادی آمد") == "هادی آمد"


def test_to_latin_digits_leaves_letters_alone():
    assert to_latin_digits("فاکتور ۱۲ كتاب") == "فاکتور 12 كتاب"


@pytest.mark.parametrize(
    "text",
    [
        "علي می خواهد ۱۲٫۵ میلیون ريال برای کتاب ها بپردازد.",
        f"{Z}  نمی دانم  {Z}",
        "حتماً\r\nفردا",
    ],
)
def test_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Spelling noise is fixed...
        ("علي كتاب", "علی کتاب"),
        ("می خواهم", f"می{Z}خواهم"),
        ("کتاب ها", f"کتاب{Z}ها"),
        ("سلام   دنیا", "سلام دنیا"),
        ("كـــتاب", "کتاب"),
        # ...but digits, separators, percent signs and diacritics stay as written,
        # because the published benchmark must test other models on them.
        ("۱۲٫۵ میلیون ريال", "۱۲٫۵ میلیون ریال"),
        ("٤٥٦ و 789 و ۱۲۳", "٤٥٦ و 789 و ۱۲۳"),
        ("۲۰٪", "۲۰٪"),
        ("حتماً", "حتماً"),
    ],
)
def test_clean_orthography(raw, expected):
    assert clean_orthography(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "علي می خواهد ۱۲٫۵ میلیون ريال برای کتاب ها بپردازد.",
        "حتماً ۲۰٪ تخفیف",
        f"{Z}  نمی دانم  {Z}",
    ],
)
def test_clean_orthography_is_idempotent_and_compatible_with_normalize(text):
    cleaned = clean_orthography(text)
    assert clean_orthography(cleaned) == cleaned
    # Cleaning published data first never changes what the model sees.
    assert normalize(cleaned) == normalize(text)
