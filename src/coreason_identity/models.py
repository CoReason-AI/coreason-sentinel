from typing import Optional, List, Any

class UserContext:
    def __init__(self, **kwargs: Any):
        self._data = kwargs

    def __getattr__(self, name: str) -> Any:
        return self._data.get(name, None)

    @property
    def sub(self) -> Optional[str]:
        return self._data.get("sub")

    @property
    def permissions(self) -> Optional[List[str]]:
        return self._data.get("permissions")
