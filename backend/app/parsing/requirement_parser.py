"""Deterministic natural-language requirement parser.

Extracts structured requirements from free-text ML problem descriptions without
any network calls. Designed behind a common interface so an LLM-backed parser
can replace it later (same output schema).

Extraction is regex + lexicon based and therefore reproducible.
"""

import re

from app.schemas import ParsedRequirements

# Country / region lexicon (extendable; matches common spellings case-insensitively).
_LOCATIONS = {
    "india", "usa", "united states", "us", "uk", "united kingdom", "canada", "germany",
    "france", "brazil", "australia", "japan", "china", "mexico", "south africa",
    "nigeria", "kenya", "europe", "africa", "asia", "latin america", "chennai", "mumbai",
    "delhi", "bangalore", "bengaluru", "hyderabad", "kolkata", "pune", "london", "paris",
    "berlin", "tokyo", "new york", "california", "texas",
}

# Domain lexicon: maps domain names to indicative keywords.
_DOMAINS = {
    "agriculture": ["crop", "crops", "yield", "farming", "agriculture", "agricultural", "harvest", "soil", "irrigation"],
    "finance": ["finance", "financial", "bank", "banking", "credit", "loan", "stock", "fraud", "insurance"],
    "healthcare": ["health", "healthcare", "medical", "disease", "patient", "hospital", "clinical", "cancer", "diabetes"],
    "transportation": ["traffic", "transport", "transportation", "congestion", "vehicle", "road", "accident", "flight"],
    "energy": ["energy", "power", "electricity", "solar", "wind", "grid", "consumption"],
    "climate": ["climate", "weather", "temperature", "rainfall", "precipitation", "drought"],
    "telecom": ["telecom", "telecommunications", "churn", "subscriber", "network"],
    "ecommerce": ["ecommerce", "e-commerce", "retail", "sales", "customer", "shopping", "purchase"],
    "education": ["education", "student", "school", "university", "learning", "enrollment"],
    "housing": ["housing", "house", "property", "real estate", "rent"],
    "employment": ["employment", "job", "salary", "wage", "labor", "labour"],
    "environment": ["environment", "pollution", "air quality", "water quality", "emission"],
}

# Task-type signal phrases.
_CLASSIFICATION_TERMS = [
    "classify", "classification", "categorize", "categorise", "predict class", "predict category",
    "predict label", "churn", "spam", "fraud detection", "detect fraud", "diagnose", "diagnosis",
    "identify type", "which category", "which class",
]
_REGRESSION_TERMS = [
    "predict amount", "predict value", "predict price", "forecast", "estimate", "regression",
    "predict yield", "predict sales", "predict demand", "how much", "how many",
]

_TARGET_HINTS = [
    "yield", "price", "revenue", "sales", "demand", "churn", "risk", "score", "rating",
    "count", "amount", "value", "cost", "temperature", "consumption",
]

_FORMATS = {
    "csv": ["csv"],
    "parquet": ["parquet"],
    "json": ["json"],
}

_YEAR = r"(?:19|20)\d{2}"


def _extract_years(text: str) -> list[int]:
    return sorted({int(m.group(0)) for m in re.finditer(rf"\b{_YEAR}\b", text) if 1900 <= int(m.group(0)) <= 2100})


def _extract_date_range(text: str) -> tuple[str | None, str | None]:
    """Handle 'between 2020 and 2025', 'from 2020 to 2025', '2020-2025', '2020 to 2025',
    'since 2020', 'after 2020', single 'in 2023'."""
    m = re.search(rf"between\s+({_YEAR})\s*(?:and|-|to|–)\s*({_YEAR})", text, re.I)
    if m:
        return m.group(1), m.group(2)
    m = re.search(rf"from\s+({_YEAR})\s*(?:to|through|till|until|-|–)\s*({_YEAR})", text, re.I)
    if m:
        return m.group(1), m.group(2)
    m = re.search(rf"({_YEAR})\s*(?:-|–|to|through)\s*({_YEAR})", text)
    if m:
        return m.group(1), m.group(2)
    m = re.search(rf"(?:since|after|from)\s+({_YEAR})", text, re.I)
    if m:
        return m.group(1), None
    m = re.search(rf"\bin\s+({_YEAR})\b", text, re.I)
    if m:
        return m.group(1), None
    years = _extract_years(text)
    if len(years) == 1:
        return str(years[0]), None
    return None, None


def _extract_locations(text: str) -> list[str]:
    found = [loc for loc in _LOCATIONS if re.search(rf"\b{re.escape(loc)}\b", text, re.I)]
    # Drop subsumed locations ("united states" when "us" also matched).
    return sorted(set(found), key=lambda s: (-len(s), s))


def _extract_domains(text: str) -> list[str]:
    domains = []
    for domain, terms in _DOMAINS.items():
        if any(re.search(rf"\b{re.escape(t)}\b", text, re.I) for t in terms):
            domains.append(domain)
    return domains


def _extract_task(text: str) -> str | None:
    if any(re.search(re.escape(t), text, re.I) for t in _CLASSIFICATION_TERMS):
        return "classification"
    if any(re.search(re.escape(t), text, re.I) for t in _REGRESSION_TERMS):
        return "regression"
    # "predict <something> price/yield/..." style phrases.
    if re.search(r"\b(predict|estimate|forecast|project)\b", text, re.I):
        if any(re.search(rf"\b{t}\b", text, re.I) for t in _TARGET_HINTS):
            return "regression"
        if re.search(r"\b(class|category|label|type)\b", text, re.I):
            return "classification"
    return None


def _extract_format(text: str) -> str | None:
    for fmt, terms in _FORMATS.items():
        if any(re.search(rf"\b{re.escape(t)}\b", text, re.I) for t in terms):
            return fmt
    return None


def _extract_row_constraints(text: str) -> tuple[int | None, int | None]:
    min_rows = max_rows = None
    m = re.search(r"(?:at least|minimum of|min(?:imum)?)\s+(?:of\s+)?([\d,]+)\s*(?:rows|records|samples)", text, re.I)
    if m:
        min_rows = int(m.group(1).replace(",", ""))
    m = re.search(r"(?:at most|maximum of|max(?:imum)?)\s+(?:of\s+)?([\d,]+)\s*(?:rows|records|samples)", text, re.I)
    if m:
        max_rows = int(m.group(1).replace(",", ""))
    return min_rows, max_rows


_STOPWORDS = {
    "find", "datasets", "dataset", "about", "between", "from", "to", "the", "and", "in", "for",
    "with", "of", "on", "a", "an", "data", "ml", "machine", "learning", "model", "models",
    "predict", "prediction", "want", "need", "looking", "show", "me", "please", "that", "can",
    "be", "used", "use", "using", "my", "our", "is", "are", "should", "would", "like", "help",
}


def _extract_keywords(text: str, domains: list[str], locations: list[str]) -> list[str]:
    """Remaining content phrases: matched domain terms + salient noun-ish tokens/bigrams."""
    keywords: list[str] = []
    for domain in domains:
        for term in _DOMAINS[domain]:
            if re.search(rf"\b{re.escape(term)}\b", text, re.I):
                keywords.append(term)
    # Bigrams of non-stopword content words, then unigrams.
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text)
    lowered = [t.lower() for t in tokens]
    known_locations = {loc.lower() for loc in locations}
    content = [
        t for t in lowered
        if t not in _STOPWORDS and t not in known_locations and not re.fullmatch(_YEAR, t)
        and len(t) > 2
    ]
    for i in range(len(content) - 1):
        bigram = f"{content[i]} {content[i + 1]}"
        if bigram in text.lower():
            keywords.append(bigram)
    for t in content:
        if not any(t in k for k in keywords):
            keywords.append(t)
    # Dedupe preserving order, cap at 10.
    seen: set[str] = set()
    result = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result[:10]


def parse_requirements(query: str) -> ParsedRequirements:
    text = query.strip()
    domains = _extract_domains(text)
    locations = _extract_locations(text)
    date_start, date_end = _extract_date_range(text)
    min_rows, max_rows = _extract_row_constraints(text)
    task = _extract_task(text)
    target_hints = [h for h in _TARGET_HINTS if re.search(rf"\b{h}\b", text, re.I)]
    return ParsedRequirements(
        keywords=_extract_keywords(text, domains, locations),
        domain=domains,
        location=locations,
        date_start=date_start,
        date_end=date_end,
        task=task,
        target_hints=target_hints[:5],
        required_features=[],
        preferred_format=_extract_format(text),
        min_rows=min_rows,
        max_rows=max_rows,
        parser="deterministic",
    )
