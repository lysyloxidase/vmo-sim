"""VMO-specific EMG electrode placement metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(slots=True)
class ElectrodePlacement:
    """Definition of a bipolar EMG electrode placement."""

    muscle: str
    orientation_degrees: float
    landmark_description: str


DEFAULT_VMO_LANDMARK: Final[str] = (
    "Electrodes centered over the distal oblique fibers of vastus medialis, "
    "approximately 4 cm superior and 3 cm medial to the superomedial patellar border, "
    "aligned with the 50-55 degree fiber direction."
)


def default_vmo_configuration() -> ElectrodePlacement:
    """Return a literature-aligned bipolar electrode placement for the VMO."""

    return ElectrodePlacement(
        muscle="VMO",
        orientation_degrees=52.5,
        landmark_description=DEFAULT_VMO_LANDMARK,
    )


__all__ = ["DEFAULT_VMO_LANDMARK", "ElectrodePlacement", "default_vmo_configuration"]
