import os
import tempfile

import pytest

# Isolated per-test-run SQLite DB; must be set before app modules import config.
_TMPDIR = tempfile.mkdtemp(prefix="automl-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMPDIR}/test.db"
os.environ["JOB_BACKEND"] = "eager"
os.environ["USE_S3_STORAGE"] = "false"

from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import Base, engine  # noqa: E402
import app.models  # noqa: F401, E402

TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def db_factory():
    return TestingSessionLocal
