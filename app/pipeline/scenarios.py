"""Load reusable BESS scenarios from the scenarios/bess/ YAML library."""

import yaml

from app.schemas.bess import BessScenario
from app.storage import Storage, get_storage

SCENARIOS_ROOT = "scenarios/bess"


def load_bess_scenario(name_or_path: str, storage: Storage | None = None) -> BessScenario:
    """Resolve `name_or_path` to a BessScenario.

    If `scenarios/bess/{name_or_path}.yaml` exists (relative to the current
    working directory, or under `storage` if given), load it from there.
    Otherwise treat `name_or_path` as a literal filesystem path.
    """
    library = storage or get_storage(".")
    candidate = f"{SCENARIOS_ROOT}/{name_or_path}.yaml"
    if library.exists(candidate):
        with library.open(candidate) as f:
            data = yaml.safe_load(f)
    else:
        with open(name_or_path, "r") as f:
            data = yaml.safe_load(f)
    return BessScenario.parse_obj(data)
