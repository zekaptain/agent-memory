"""Human-trained admission, separate from the read-only judge experiments."""

from ..db import AgentMemoryError

ADMISSION_VERSION = "admission_v1"


class AdmissionError(AgentMemoryError):
    """Admission cannot safely proceed with the supplied configuration or files."""
