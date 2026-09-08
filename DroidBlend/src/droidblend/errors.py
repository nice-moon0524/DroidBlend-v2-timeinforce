class DroidBlendError(RuntimeError):
    """Base error raised for an invalid experiment state."""


class ConfigurationError(DroidBlendError):
    """Raised when a YAML experiment configuration is incomplete or invalid."""


class CompatibilityError(DroidBlendError):
    """Raised when sender and receiver cannot share a DroidSpeak cache."""


class ArtifactError(DroidBlendError):
    """Raised when a cache/profile artifact is missing or incompatible."""
