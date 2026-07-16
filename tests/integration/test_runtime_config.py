from pathlib import Path

import pytest
import yaml

from werewolf.runtime_config import validate_runtime_config


def test_canonical_collection_config_is_strict():
    config = yaml.safe_load(Path("configs/tom/collect.yaml").read_text(encoding="utf-8"))
    assert validate_runtime_config(config)
    legacy = {"backend": {}, "agent_config": {}, "env_config": {}}
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_runtime_config(legacy)
