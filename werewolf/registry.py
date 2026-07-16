"""Small explicit class registry used by gameplay agents."""


class Registry:
    def __init__(self, name):
        self.name = name
        self.entries = {}

    def register(self, keys):
        def decorator(cls):
            for key in keys:
                if key in self.entries and self.entries[key] is not cls:
                    raise ValueError(f"{key!r} is already registered in {self.name}")
                self.entries[key] = cls
            return cls
        return decorator

    def create(self, key, **kwargs):
        try:
            cls = self.entries[key]
        except KeyError as exc:
            raise ValueError(
                f"unknown {self.name} type {key!r}; available={sorted(self.entries)}"
            ) from exc
        return cls(**kwargs)

    def get_all_entries(self):
        return dict(self.entries)
