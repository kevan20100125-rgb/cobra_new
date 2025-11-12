"""
overwatch.py

Utility class for creating a centralized/standardized logger (built on Rich) and accelerate handler.
"""
import logging
import logging.config
import os
import time
from contextlib import contextmanager
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Tuple, Union

# Overwatch Default Format String
RICH_FORMATTER, DATEFMT = "| >> %(message)s", "%m/%d [%H:%M:%S]"

# Set Logging Configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT}},
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "formatter": "simple-console",
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOG_CONFIG)


# === Custom Contextual Logging Logic ===
class ContextAdapter(LoggerAdapter):
    CTX_PREFIXES: ClassVar[Dict[int, str]] = {**{0: "[*] "}, **{idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]}}

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Tuple[str, MutableMapping[str, Any]]:
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


class _ScopedLoggerMixin:
    def __init__(self, logger: ContextAdapter) -> None:
        self._scope_depth = 0
        self._scoped_logger = logger

    @contextmanager
    def scoped(self, message: str) -> Any:
        level = self._scope_depth
        self._scoped_logger.info(f"{message} (start)", ctx_level=level)
        self._scope_depth += 1
        start = time.time()
        try:
            yield
        finally:
            self._scope_depth = max(0, self._scope_depth - 1)
            duration = time.time() - start
            self._scoped_logger.info(f"{message} (done in {duration:.2f}s, depth={level})", ctx_level=level)


class DistributedOverwatch(_ScopedLoggerMixin):
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that wraps logging & `accelerate.PartialState`."""
        from accelerate import PartialState

        # Note that PartialState is always safe to initialize regardless of `accelerate launch` or `torchrun`
        #   =>> However, might be worth actually figuring out if we need the `accelerate` dependency at all!
        self.logger, self.distributed_state = ContextAdapter(logging.getLogger(name), extra={}), PartialState()
        super().__init__(self.logger)

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> only Log `INFO` on Main Process, `ERROR` on others!
        self.logger.setLevel(logging.INFO if self.distributed_state.is_main_process else logging.ERROR)

    def rank_zero_only(self) -> Callable[..., Any]:
        return self.distributed_state.on_main_process

    def is_rank_zero(self) -> bool:
        return self.distributed_state.is_main_process

    def rank(self) -> int:
        return self.distributed_state.process_index

    def world_size(self) -> int:
        return self.distributed_state.num_processes


class PureOverwatch(_ScopedLoggerMixin):
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that just wraps logging."""
        self.logger = ContextAdapter(logging.getLogger(name), extra={})
        super().__init__(self.logger)

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> INFO
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def rank_zero_only() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return identity

    @staticmethod
    def is_rank_zero() -> bool:
        return True

    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def world_size() -> int:
        return 1


def initialize_overwatch(name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    return DistributedOverwatch(name) if int(os.environ.get("WORLD_SIZE", -1)) != -1 else PureOverwatch(name)


def finalize_summary(manifest: Dict[str, Any], verify_report: Dict[str, Any]) -> str:
    """
    Produce a compact textual summary after finalize/export step.
    """
    items = manifest.get("items", [])
    lines = [
        f"[Finalize] modules={len(items)}",
        f"[Finalize] verify_status={verify_report.get('status')}",
        f"[Finalize] verify_batches={verify_report.get('batches', 0)} "
        f"max_mae={verify_report.get('max_mae', 0):.4f} max_rel={verify_report.get('max_rel', 0):.4f}",
    ]
    violations = verify_report.get("violations") or []
    if violations:
        lines.append("[Finalize] top layer deviations:")
        for mae, name in violations[:5]:
            lines.append(f"  - {name}: MAE={mae:.4f}")
    return "\n".join(lines)
