from typing import Final

from matplotlib.style import context

from ._internal.constants import Constant


class DefaultThemes(metaclass=Constant):
    """
    The default themes that are included with matplotlib.
    """

    SOLARIZED_LIGHT: Final[str] = "Solarize_Light2"
    MPL_GALLERY: Final[str] = "_mpl-gallery"
    MPL_GALLERY_NOGRID: Final[str] = "_mpl-gallery-nogrid"
    BMH: Final[str] = "bmh"
    CLASSIC: Final[str] = "classic"
    DARK_BACKGROUND: Final[str] = "dark_background"
    FAST: Final[str] = "fast"
    FIVETHIRTYEIGHT: Final[str] = "fivethirtyeight"
    GGPLOT: Final[str] = "ggplot"
    GRAYSCALE: Final[str] = "grayscale"
    PETROFF10: Final[str] = "petroff10"
    SEABORN: Final[str] = "seaborn-v0_8"
    SEABORN_BRIGHT: Final[str] = "seaborn-v0_8-bright"
    SEABORN_COLORBLIND: Final[str] = "seaborn-v0_8-colorblind"
    SEABORN_DARK: Final[str] = "seaborn-v0_8-dark"
    SEABORN_DARK_PALETTE: Final[str] = "seaborn-v0_8-dark-palette"
    SEABORN_DARKGRID: Final[str] = "seaborn-v0_8-darkgrid"
    SEABORN_DEEP: Final[str] = "seaborn-v0_8-deep"
    SEABORN_MUTED: Final[str] = "seaborn-v0_8-muted"
    SEABORN_NOTEBOOK: Final[str] = "seaborn-v0_8-notebook"
    SEABORN_PAPER: Final[str] = "seaborn-v0_8-paper"
    SEABORN_PASTEL: Final[str] = "seaborn-v0_8-pastel"
    SEABORN_POSTER: Final[str] = "seaborn-v0_8-poster"
    SEABORN_TALK: Final[str] = "seaborn-v0_8-talk"
    SEABORN_TICKS: Final[str] = "seaborn-v0_8-ticks"
    SEABORN_WHITE: Final[str] = "seaborn-v0_8-white"
    SEABORN_WHITEGRID: Final[str] = "seaborn-v0_8-whitegrid"
    TABLEAU_COLORBLIND: Final[str] = "tableau-colorblind10"

    available: Final[tuple[str, ...]] = (
        "SOLARIZED_LIGHT",
        "MPL_GALLERY",
        "MPL_GALLERY_NOGRID",
        "BMH",
        "CLASSIC",
        "DARK_BACKGROUND",
        "FAST",
        "FIVETHIRTYEIGHT",
        "GGPLOT",
        "GRAYSCALE",
        "PETROFF10",
        "SEABORN",
        "SEABORN_BRIGHT",
        "SEABORN_COLORBLIND",
        "SEABORN_DARK",
        "SEABORN_DARK_PALETTE",
        "SEABORN_DARKGRID",
        "SEABORN_DEEP",
        "SEABORN_MUTED",
        "SEABORN_NOTEBOOK",
        "SEABORN_PAPER",
        "SEABORN_PASTEL",
        "SEABORN_POSTER",
        "SEABORN_TALK",
        "SEABORN_TICKS",
        "SEABORN_WHITE",
        "SEABORN_WHITEGRID",
        "TABLEAU_COLORBLIND",
    )


__all__: list[str] = ["DefaultThemes", "context"]
