"""Location expansion helpers shared by search and ranking."""

from app.schemas import ParsedRequirements

# A location plus progressively broader regions (chennai -> tamil nadu -> india).
LOCATION_PARENTS = {
    "chennai": "tamil nadu", "mumbai": "maharashtra", "delhi": "india",
    "bangalore": "karnataka", "bengaluru": "karnataka", "hyderabad": "telangana",
    "kolkata": "west bengal", "pune": "maharashtra",
    "california": "usa", "new york": "usa", "london": "uk",
    "tamil nadu": "india", "maharashtra": "india", "karnataka": "india",
    "telangana": "india", "west bengal": "india", "berlin": "germany", "paris": "france",
}


def location_variants(req: ParsedRequirements) -> list[str]:
    variants: list[str] = []
    for loc in req.location:
        loc_l = loc.lower()
        if loc_l not in variants:
            variants.append(loc_l)
        parent = LOCATION_PARENTS.get(loc_l)
        if parent and parent not in variants:
            variants.append(parent)
    return variants
