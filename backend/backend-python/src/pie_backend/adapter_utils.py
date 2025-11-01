"""Utility for safely importing adapter functionality with fallback."""

from typing import TYPE_CHECKING

# Safe import of AdapterSubpass with fallback
try:
    from adapter import AdapterSubpass  # type: ignore[import]
except ImportError:
    # AdapterSubpass will be provided by the backend - use object as fallback
    AdapterSubpass = object  # type: ignore[misc]

# Type definitions for when AdapterSubpass is not available
if TYPE_CHECKING:
    try:
        from adapter import AdapterSubpass  # type: ignore[import]
    except ImportError:
        # Define a stub for type checking
        class AdapterSubpass:  # type: ignore[misc]  # pylint: disable=function-redefined
            """Type stub for AdapterSubpass when not available."""

            def __init__(self, **kwargs) -> None:  # pylint: disable=unused-argument
                """Initialize adapter subpass."""

            def execute(self, *args, **kwargs):  # pylint: disable=unused-argument
                """Execute adapter subpass."""


def get_adapter_subpass():
    """Get the AdapterSubpass class, or None if not available."""
    return None if AdapterSubpass is object else AdapterSubpass


def ensure_adapter_available():
    """Ensure AdapterSubpass is available, raising an error if not."""
    if AdapterSubpass is object:
        raise RuntimeError(
            "Adapter functionality is required but AdapterSubpass is not available. "
            "Please ensure the adapter module is properly imported."
        )
    return AdapterSubpass
