"""Dataset loader for successful single-protocol ``tom.v1_1`` records."""

from copy import deepcopy
import json
from pathlib import Path

from torch.utils.data import Dataset

from werewolf.tom.collection import assert_audit_passes
from werewolf.tom.features import sample_to_features
from werewolf.tom.schemas import validate_sample, validate_sample_collection


def _run_files_for_samples(path):
    suffix = ".samples.jsonl"
    if not path.name.endswith(suffix):
        raise ValueError(
            f"{path}: collection samples must use <run_id>.samples.jsonl"
        )
    run_id = path.name[:-len(suffix)]
    return (
        path.with_name(f"{run_id}.audit.json"),
        path.with_name(f"{run_id}.failures.jsonl"),
    )


def _validate_audit_metadata(path, audit, records):
    protocols = [record["prompt_protocol"] for record in records]
    expected = {
        "games": len({record["game_id"] for record in records}),
        "prompt_protocol_ids": sorted(
            {protocol["protocol_id"] for protocol in protocols}
        ),
        "gameplay_prompt_versions": sorted(
            {protocol["gameplay"]["version"] for protocol in protocols}
        ),
        "belief_prompt_versions": sorted(
            {protocol["belief"]["version"] for protocol in protocols}
        ),
        "parser_prompt_versions": sorted(
            {protocol["parser"]["version"] for protocol in protocols}
        ),
        "gameplay_prompt_hashes": sorted(
            {protocol["gameplay"]["sha256"] for protocol in protocols}
        ),
        "belief_prompt_hashes": sorted(
            {protocol["belief"]["sha256"] for protocol in protocols}
        ),
        "parser_prompt_hashes": sorted(
            {protocol["parser"]["sha256"] for protocol in protocols}
        ),
        "ruleset_ids": sorted({protocol["ruleset"]["id"] for protocol in protocols}),
        "ruleset_versions": sorted(
            {protocol["ruleset"]["version"] for protocol in protocols}
        ),
        "ruleset_hashes": sorted(
            {protocol["ruleset"]["sha256"] for protocol in protocols}
        ),
    }
    mismatched = {
        field: {"audit": audit.get(field), "samples": value}
        for field, value in expected.items()
        if audit.get(field) != value
    }
    if mismatched:
        raise ValueError(f"{path}: collection audit metadata mismatch: {mismatched}")


class ToMDataset(Dataset):
    def __init__(
        self,
        paths,
        *,
        task=None,
        mode=None,
        include_first_order_private=True,
    ):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        records = []
        self.include_first_order_private = include_first_order_private
        for path in paths:
            path = Path(path)
            audit_path, failures_path = _run_files_for_samples(path)
            if not audit_path.is_file():
                raise ValueError(f"{path}: missing collection audit {audit_path}")
            if not failures_path.is_file():
                raise ValueError(f"{path}: missing collection failures {failures_path}")
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{audit_path}: invalid collection audit") from exc
            if audit.get("schema_version") != "tom.audit.v1_4":
                raise ValueError(f"{audit_path}: unsupported collection audit")
            if (
                audit.get("collection_status") != "complete"
                or audit.get("runtime_failure_count") != 0
            ):
                raise ValueError(f"{path}: failed collection audit is not trainable")
            try:
                assert_audit_passes(audit)
            except RuntimeError as exc:
                raise ValueError(f"{audit_path}: invalid collection audit") from exc
            path_records = []
            with path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        validate_sample(record)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
                    path_records.append(record)
            _validate_audit_metadata(path, audit, path_records)
            records.extend(path_records)
        try:
            validate_sample_collection(records)
        except ValueError as exc:
            raise ValueError(f"dataset identity/alignment error: {exc}") from exc
        protocol_ids = sorted(
            {record["prompt_protocol"]["protocol_id"] for record in records}
        )
        if len(protocol_ids) != 1:
            raise ValueError(
                f"dataset must contain one prompt protocol; found={protocol_ids}"
            )
        self.protocol_id = protocol_ids[0]
        self.prompt_protocol = deepcopy(records[0]["prompt_protocol"])
        self.records = [
            record
            for record in records
            if (task is None or record["task"] == task)
            and (mode is None or record["mode"] == mode)
        ]
        if not self.records:
            raise ValueError("dataset contains no matching successful tom.v1_1 samples")
        self.game_ids = frozenset(record["game_id"] for record in self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return sample_to_features(
            self.records[index],
            include_first_order_private=self.include_first_order_private,
        )
