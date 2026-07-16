import json
import os

from werewolf.helper.log_utils import Log
from werewolf.models.twd_tom.samples import make_twd_tom_sample


def _json_default(obj):
    if isinstance(obj, Log):
        return obj.__dict__
    return json.JSONEncoder().default(obj)


class TWDToMSampleCollector:
    def __init__(self, output_path: str, game_id=None):
        parent_directory = os.path.dirname(
            os.path.abspath(output_path)
        )
        os.makedirs(parent_directory, exist_ok=True)
        self.output_path = output_path
        self.game_id = game_id
        self._file = open(
            output_path,
            "a",
            encoding="utf-8",
        )

    def record(
        self,
        observation: dict,
        roles,
        step_idx=None,
        alive_mask=None,
    ) -> dict:
        sample = make_twd_tom_sample(
            observation=observation,
            roles=roles,
            game_id=self.game_id,
            alive_mask=alive_mask,
        )
        sample["step_idx"] = step_idx
        line = json.dumps(
            sample,
            ensure_ascii=False,
            default=_json_default,
        )
        self._file.write(line + "\n")
        self._file.flush()
        return sample

    def close(self):
        if not self._file.closed:
            self._file.close()
