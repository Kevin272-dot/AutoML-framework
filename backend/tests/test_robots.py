from app.discovery.robots import evaluate_robots


def test_robots_found_allowed():
    txt = """
User-agent: *
Disallow: /private/
"""
    result = evaluate_robots(txt, 200, "/api/datasets?search=x")
    assert result.status == "FOUND"
    assert result.allowed is True


def test_robots_found_disallowed():
    txt = """
User-agent: *
Disallow: /
"""
    result = evaluate_robots(txt, 200, "/api/datasets")
    assert result.status == "FOUND"
    assert result.allowed is False
    assert "/" in result.disallowed_paths


def test_robots_not_found():
    result = evaluate_robots(None, 404, "/anything")
    assert result.status == "NOT_FOUND"
    assert result.allowed is True


def test_robots_error():
    result = evaluate_robots(None, None, "/anything")
    assert result.status == "ERROR"
    assert result.allowed is None
