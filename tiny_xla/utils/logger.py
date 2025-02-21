from __future__ import annotations

import contextlib
import dataclasses
import enum
import logging
import sys
from pathlib import Path
from typing import Any, Final, Iterator


@dataclasses.dataclass(frozen=True, slots=True)
class LogConfig:
    """Configuration for the logger."""

    level: int = logging.INFO
    log_file: Path | None = None
    show_colors: bool = True
    show_line_numbers: bool = True
    show_timestamps: bool = True


class LogColor(enum.StrEnum):
    """ANSI color codes for terminal output."""

    GREY = "\033[38;20m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_RED = "\033[31;1m"
    PURPLE = "\033[95m"
    RESET = "\033[0m"


class Formatter(logging.Formatter):
    """Custom formatter for log messages."""

    FORMATS: Final[dict[int, str]] = {
        logging.DEBUG: LogColor.GREY,
        logging.INFO: LogColor.GREEN,
        logging.WARNING: LogColor.YELLOW,
        logging.ERROR: LogColor.RED,
        logging.CRITICAL: LogColor.BOLD_RED,
    }

    def __init__(
        self,
        show_colors: bool = True,
        show_line_numbers: bool = True,
        show_timestamps: bool = True,
    ) -> None:
        self.show_colors = show_colors
        self.show_line_numbers = show_line_numbers
        self.show_timestamps = show_timestamps

        fmt_parts: list[str] = []
        if show_timestamps:
            fmt_parts.append("%(asctime)s")
        if show_timestamps:
            fmt_parts.append("%(filename)s:%(lineno)d")
        fmt_parts.extend(["%(levelname)s", "%(name)s", "%(message)s"])

        super().__init__(
            fmt=" - ".join(fmt_parts), datefmt="%Y-%m-%d %H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        if not self.show_colors():
            return super().format(record)

        color = self.FORMATS.get(record.levelno, LogColor.GREY)
        record.levelname = f"{color}{record.levelname}{LogColor.RESET}"
        record.msg = f"{color}{record.msg}{LogColor.RESET}"
        return super().format(record)


class XLALogger:
    _loggers: dict[str, logging.Logger] = {}
    _config: LogConfig | None = None

    @classmethod
    def configure(cls, config: LogConfig) -> None:
        """Configure global logging settings."""
        cls._config = config

        # Reset existing loggers
        for logger in cls._loggers.values():
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

        # Set up handlers based on config
        handlers: list[logging.Handler] = []

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            Formatter(
                show_colors=config.show_colors,
                show_line_numbers=config.show_line_numbers,
                show_timestamps=config.show_timestamps,
            )
        )
        handlers.append(console)

        # Optional file handler
        if config.log_file is not None:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setFormatter(
                Formatter(
                    show_colors=False,
                    show_line_numbers=config.show_line_numbers,
                    show_timestamps=config.show_timestamps,
                )
            )
            handlers.append(file_handler)

        # Update all existing loggers
        for logger in cls._loggers.values():
            logger.setLevel(config.level)
            for handler in handlers:
                logger.addHandler(handler)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger with the given name."""
        if name not in cls._loggers:
            logger = logging.getLogger(name)

            if cls._config is not None:
                logger.setLevel(cls._config.level)
                cls.configure(cls._config)

            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    @contextlib.contextmanager
    def phase(
        cls, name: str, logger_name: str | None = None
    ) -> Iterator[None]:
        """Context manager for logging phases."""
        logger = cls.get_logger(logger_name or "tiny_xla.compiler")
        try:
            logger.info("Starting phase: %s", name)
            yield
        except Exception as e:
            logger.error("Phase %s failed: %s", name, e)
            raise
        else:
            logger.info("Completed phase: %s", name)

    @classmethod
    @contextlib.contextmanager
    def scoped_debug(
        cls, logger_name: str, scope_name: str, **context: Any
    ) -> Iterator[None]:
        """Context manager for detailed debug logging with context."""
        logger = cls.get_logger(logger_name)
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        try:
            logger.debug("Entering %s (%s)", scope_name, context_str)
            yield
        finally:
            logger.debug("Exiting %s", scope_name)


def setup_default_logger() -> None:
    """Set up the logger configuration."""
    XLALogger.configure(LogConfig())


setup_default_logger()
