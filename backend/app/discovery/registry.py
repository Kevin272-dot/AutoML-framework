"""Seed registry of known data sources.

Resource discovery = matching the parsed requirements against this registry
(domain/keyword coverage). No network access happens during discovery; auditing
happens separately and is still followed by explicit user approval.
"""

from sqlalchemy.orm import Session

from app.models import DataSource

KNOWN_SOURCES: list[dict] = [
    {
        "slug": "huggingface",
        "name": "Hugging Face Datasets",
        "base_url": "https://huggingface.co",
        "source_type": "DATASET_CATALOG",
        "api_url": "https://huggingface.co/api/datasets",
        "robots_url": "https://huggingface.co/robots.txt",
        "terms_url": "https://huggingface.co/terms-of-service",
        "license_url": None,
        "access_method": "API",
        "domains": [
            "agriculture", "finance", "healthcare", "transportation", "energy", "climate",
            "telecom", "ecommerce", "education", "housing", "employment", "environment",
        ],
        "supports_search": True,
        "supports_metadata": True,
        "supports_preview": True,
        "supports_download": True,
        "supports_api": True,
        "requires_auth": False,
        "adapter": "huggingface",
    },
    {
        "slug": "kaggle",
        "name": "Kaggle Datasets",
        "base_url": "https://www.kaggle.com",
        "source_type": "DATASET_CATALOG",
        "api_url": "https://www.kaggle.com/api/v1/datasets/list",
        "robots_url": "https://www.kaggle.com/robots.txt",
        "terms_url": "https://www.kaggle.com/terms",
        "license_url": None,
        "access_method": "API",
        "domains": [
            "agriculture", "finance", "healthcare", "transportation", "energy", "climate",
            "telecom", "ecommerce", "education", "housing", "employment",
        ],
        "supports_search": True,
        "supports_metadata": True,
        "supports_preview": False,
        "supports_download": True,
        "supports_api": True,
        "requires_auth": True,  # Kaggle API requires credentials
        "adapter": "kaggle",  # adapter not implemented yet (Phase 12)
    },
    {
        "slug": "uci",
        "name": "UCI Machine Learning Repository",
        "base_url": "https://archive.ics.uci.edu",
        "source_type": "REPOSITORY",
        "api_url": None,
        "robots_url": "https://archive.ics.uci.edu/robots.txt",
        "terms_url": None,
        "license_url": None,
        "access_method": "DOWNLOAD",
        "domains": ["healthcare", "finance", "education", "transportation", "environment"],
        "supports_search": True,
        "supports_metadata": True,
        "supports_preview": False,
        "supports_download": True,
        "supports_api": False,
        "requires_auth": False,
        "adapter": None,
    },
    {
        "slug": "data-gov-in",
        "name": "data.gov.in (India Open Government Data)",
        "base_url": "https://www.data.gov.in",
        "source_type": "OPEN_DATA_PORTAL",
        "api_url": "https://www.data.gov.in/apis",
        "robots_url": "https://www.data.gov.in/robots.txt",
        "terms_url": "https://www.data.gov.in/terms-and-conditions",
        "license_url": None,
        "access_method": "API",
        "domains": ["agriculture", "healthcare", "education", "transportation", "energy", "employment", "housing"],
        "supports_search": True,
        "supports_metadata": True,
        "supports_preview": False,
        "supports_download": True,
        "supports_api": True,
        "requires_auth": True,  # data.gov.in API keys
        "adapter": None,
    },
    {
        "slug": "data-gov",
        "name": "data.gov (US Open Government Data)",
        "base_url": "https://data.gov",
        "source_type": "OPEN_DATA_PORTAL",
        "api_url": "https://catalog.data.gov/api/3",
        "robots_url": "https://data.gov/robots.txt",
        "terms_url": "https://www.data.gov/terms-of-use",
        "license_url": None,
        "access_method": "API",
        "domains": ["agriculture", "finance", "healthcare", "transportation", "energy", "climate", "education", "environment"],
        "supports_search": True,
        "supports_metadata": True,
        "supports_preview": False,
        "supports_download": True,
        "supports_api": True,
        "requires_auth": False,
        "adapter": None,
    },
    {
        "slug": "eu-open-data",
        "name": "EU Open Data Portal",
        "base_url": "https://data.europa.eu",
        "source_type": "OPEN_DATA_PORTAL",
        "api_url": "https://data.europa.eu/api/hub/search",
        "robots_url": "https://data.europa.eu/robots.txt",
        "terms_url": None,
        "license_url": None,
        "access_method": "API",
        "domains": ["agriculture", "finance", "healthcare", "transportation", "energy", "climate", "environment", "employment"],
        "supports_search": True,
        "supports_metadata": True,
        "supports_preview": False,
        "supports_download": True,
        "supports_api": True,
        "requires_auth": False,
        "adapter": None,
    },
]


def seed_sources(db: Session) -> None:
    """Idempotently insert known sources."""
    existing = {slug for (slug,) in db.query(DataSource.slug).all()}
    for spec in KNOWN_SOURCES:
        if spec["slug"] in existing:
            continue
        db.add(DataSource(id=_new_id(), **spec))
    db.commit()


def _new_id() -> str:
    import uuid

    return str(uuid.uuid4())
