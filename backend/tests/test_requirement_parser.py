from app.parsing.requirement_parser import parse_requirements


def test_spec_example_crop_yield():
    req = parse_requirements("Find datasets about crop yield in India between 2020 and 2025")
    assert "india" in req.location
    assert "agriculture" in req.domain
    assert req.date_start == "2020"
    assert req.date_end == "2025"
    assert any("crop" in kw or "yield" in kw for kw in req.keywords)


def test_from_to_range():
    req = parse_requirements("telecom customer churn data from 2019 to 2023")
    assert (req.date_start, req.date_end) == ("2019", "2023")
    assert "telecom" in req.domain
    assert "churn" in " ".join(req.keywords)


def test_hyphen_range():
    req = parse_requirements("traffic congestion chennai 2023-2025")
    assert (req.date_start, req.date_end) == ("2023", "2025")
    assert "transportation" in req.domain
    assert any("chennai" in loc for loc in req.location)


def test_classification_task_detection():
    req = parse_requirements("datasets to classify credit card fraud detection")
    assert req.task == "classification"


def test_regression_task_detection():
    req = parse_requirements("predict housing price for california")
    assert req.task == "regression"
    assert "housing" in req.domain
    assert "california" in req.location


def test_row_constraints_and_format():
    req = parse_requirements("sales data at least 5000 rows in csv format since 2021")
    assert req.min_rows == 5000
    assert req.preferred_format == "csv"
    assert req.date_start == "2021"
    assert req.date_end is None


def test_single_year():
    req = parse_requirements("energy consumption data in 2022")
    assert req.date_start == "2022"


def test_empty_constraints_get_neutral_defaults():
    req = parse_requirements("something completely unrelated xyzzy")
    assert req.domain == []
    assert req.date_start is None
    assert req.parser == "deterministic"
