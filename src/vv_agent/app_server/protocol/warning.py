from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WarningParams:
    message: str
    code: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {"message": self.message}
        if self.code is not None:
            payload["code"] = self.code
        return payload
