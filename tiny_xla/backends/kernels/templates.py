from __future__ import annotations

import os
import string
from pathlib import Path


class KernelTemplate:
    """Manages kernel templates and substitution."""

    def __init__(self, template_str: str) -> None:
        self.template = string.Template(template_str)

    def substitute(self, **kwargs) -> str:
        """Substitute variables in template."""
        return self.template.substitute(**kwargs)


class KernelLibrary:
    """Manages kernel template loading and caching."""

    def __init__(self, kernel_dir: str | None = None) -> None:
        if kernel_dir is None:
            kernel_dir = os.path.join(os.path.dirname(__file__), "cpu")
        self.kernel_dir = Path(kernel_dir)
        self._templates: dict[str, KernelTemplate] = {}

    def get_template(self, name: str) -> KernelTemplate:
        """Get kernel template by name."""
        if name not in self._templates:
            template_path = self.kernel_dir / f"{name}.c"
            if not template_path.exists():
                raise ValueError(f"Kernel template {name} not found.")

            with open(template_path, "r") as f:
                self._templates[name] = KernelTemplate(f.read())

        return self._templates[name]


# Global kernel library instance
KERNEL_LIBRARY = KernelLibrary()
