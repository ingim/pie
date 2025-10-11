"""
Base shader compiler with Metal file utilities.

This module provides the base class for compiling Metal shaders
and common utilities for reading and processing Metal source files.
"""

from pathlib import Path
from typing import Dict, Any
from .mps_config import MPS_COMPILE_AVAILABLE, MPS_DEVICE_AVAILABLE, compile_shader


class BaseShaderCompiler:
    """Base class for Metal shader compilation and management."""

    def __init__(self):
        self.compiled_libraries: Dict[str, Any] = {}
        self.kernel_dir = Path(__file__).parent / "metal" / "kernels"

    def can_use_mps_kernels(self) -> bool:
        """Check if we can use compiled MPS kernels."""
        return (MPS_COMPILE_AVAILABLE and
                MPS_DEVICE_AVAILABLE and
                len(self.compiled_libraries) > 0)

    def _read_metal_file(self, filename: str) -> str:
        """Read Metal kernel source file."""
        file_path = self.kernel_dir / filename
        if file_path.exists():
            return file_path.read_text()
        else:
            print(f"⚠️  Metal file not found: {filename}")
            return ""

    def _process_common_header(self, common_source: str) -> str:
        """Process the common header to resolve any includes."""
        # Remove the duplicate #include <metal_stdlib> and using namespace since
        # they'll be included in the final shader source
        processed = common_source.replace('#include <metal_stdlib>\n', '')
        processed = processed.replace('using namespace metal;\n', '')

        # Remove empty lines at the beginning
        lines = processed.split('\n')
        while lines and lines[0].strip() == '':
            lines.pop(0)

        return '\n'.join(lines)

    def _resolve_includes(self, source: str, common_source: str) -> str:
        """Resolve includes in Metal source code."""
        lines = source.split('\n')
        resolved_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#include "metal_attention_common.metal"'):
                # Skip this include since we're already including the common source
                resolved_lines.append('// Resolved: #include "metal_attention_common.metal"')
            elif stripped.startswith('#include <metal_stdlib>'):
                # Skip duplicate metal_stdlib includes
                resolved_lines.append('// Skipped duplicate: #include <metal_stdlib>')
            elif stripped.startswith('using namespace metal;'):
                # Skip duplicate namespace declarations
                resolved_lines.append('// Skipped duplicate: using namespace metal;')
            else:
                resolved_lines.append(line)

        return '\n'.join(resolved_lines)

    def _compile_shader(self, source: str, library_name: str) -> bool:
        """
        Compile a Metal shader and store it in compiled_libraries.

        Returns True if compilation succeeded, False otherwise.
        """
        if not MPS_COMPILE_AVAILABLE:
            return False

        try:
            self.compiled_libraries[library_name] = compile_shader(source)
            return True
        except Exception as e:
            print(f"❌ Failed to compile {library_name} shader: {e}")
            return False