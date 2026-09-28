from pathlib import Path
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def save_fig(name):
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
