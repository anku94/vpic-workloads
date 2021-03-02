import pickle
from pathlib import Path
from typing import Dict, Any, Union


class Cache:
    cache_path: Path
    cache: Dict[str, bytes]

    def __init__(self):
        self.cache = {}
        self.cache_path = Path('./.pyperfcache')
        self.load_cache()

    def __del__(self):
        self.save_cache()

    def load_cache(self) -> None:
        if self.cache_path.exists():
            with self.cache_path.open('rb') as f:
                try:
                    self.cache = pickle.loads(f.read())
                except Exception as e:
                    self.cache = {}

    def save_cache(self) -> None:
        with self.cache_path.open('wb') as f:
            data = pickle.dumps(self.cache)
            f.write(data)

    def put(self, key: str, val: object) -> None:
        obj_str = pickle.dumps(val)
        self.cache[key] = obj_str

    def get(self, key: str) -> object:
        if key in self.cache:
            return pickle.loads(self.cache[key])
        else:
            return None

    def exists(self, key: str) -> bool:
        return True if key in self.cache else False

