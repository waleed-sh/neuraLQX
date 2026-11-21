"""Regenerate the original SVG figures in the distributed simulation guides."""

from html import escape
from pathlib import Path

DESTINATION = Path(__file__).resolve().parents[1] / "assets/images/distributed"
INK = "#18363b"
TEAL = "#006d77"
SEAFOAM = "#83c5be"
ICE = "#edf6f9"
PEACH = "#ffddd2"
TERRA = "#e29578"
COLORS = [TEAL, SEAFOAM, TERRA, PEACH]


class Figure:
    def __init__(self, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{height}" '
            f'viewBox="0 0 800 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" '
            'refX="7" refY="4" orient="auto-start-reverse">'
            f'<path d="M0 0 L8 4 L0 8 Z" fill="{TEAL}"/></marker></defs>',
            f'<g font-family="Arial, Helvetica, sans-serif" fill="{INK}" font-size="21">',
        ]
        self.rect(0, 0, 800, height, ICE, radius=18)

    def rect(self, x, y, width, height, fill="white", stroke="none", radius=8):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

    def text(self, x, y, value, size=21, color=INK, weight="normal", anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'
        )

    def line(self, x1, y1, x2, y2, arrow=False):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{TEAL}" '
            f'stroke-width="2.5" fill="none"{marker}/>'
        )

    def save(self, name):
        DESTINATION.mkdir(parents=True, exist_ok=True)
        (DESTINATION / name).write_text("\n".join([*self.parts, "</g></svg>\n"]))


def sample_sharding():
    fig = Figure(
        600,
        "One global array, four local sample shards",
        "Eight rows of four configuration labels are split in pairs over four devices. "
        "Each device holds all model parameters and complete four-label configurations.",
    )
    fig.text(32, 48, "One global array, four local shards", 29, weight="bold")
    fig.text(32, 83, "Partition rows. Keep every configuration complete.", 22)
    fig.text(155, 145, "Global shape (8, 4)", 22, anchor="middle")
    for row in range(8):
        for col in range(4):
            fig.rect(44 + col * 54, 170 + row * 34, 48, 28, COLORS[row // 2], radius=3)
        fig.text(25, 192 + row * 34, str(row), 17, anchor="middle")
    fig.text(155, 475, "4 labels per configuration", 19, anchor="middle")
    fig.line(277, 300, 339, 300, arrow=True)
    for device in range(4):
        x = 365 + (device % 2) * 212
        y = 125 + (device // 2) * 205
        fig.rect(x, y, 192, 184, stroke=COLORS[device])
        fig.text(x + 96, y + 31, f"Device {device}", 22, weight="bold", anchor="middle")
        fig.text(x + 96, y + 59, f"Rows {device * 2} and {device * 2 + 1}", 19, anchor="middle")
        for row in range(2):
            for col in range(4):
                fig.rect(x + 22 + col * 38, y + 76 + row * 27, 32, 21, COLORS[device], radius=3)
        fig.rect(x + 15, y + 138, 162, 31, ICE, radius=5)
        fig.text(x + 96, y + 160, "All parameters θ", 18, anchor="middle")
    fig.text(400, 571, 'Samples: P("samples", None)     Parameters: P()', 21, anchor="middle")
    fig.save("sample-sharding.svg")


def processes():
    fig = Figure(
        560,
        "Two hosts, two processes, eight global devices",
        "Each of two hosts runs one Python process controlling four GPUs. "
        "The processes cooperate in one global calculation, each owning four local shards.",
    )
    fig.text(32, 48, "Two hosts, one global calculation", 29, weight="bold")
    fig.text(32, 82, "2 Python processes · 4 local GPUs each · 8 devices in total", 21)
    for host in range(2):
        x = 24 + host * 390
        fig.rect(x, 114, 362, 345, stroke=TEAL)
        fig.text(x + 181, 148, f"Host {host}", 25, weight="bold", anchor="middle")
        fig.rect(x + 23, 168, 316, 50, TEAL)
        fig.text(x + 181, 200, f"Python process {host}", 23, color="white", anchor="middle")
        for local in range(4):
            gx = x + 23 + (local % 2) * 164
            gy = 245 + (local // 2) * 98
            fig.rect(gx, gy, 152, 76, SEAFOAM if host == 0 else PEACH)
            fig.text(
                gx + 76, gy + 30, f"GPU {host * 4 + local}", 23, weight="bold", anchor="middle"
            )
            fig.text(gx + 76, gy + 57, "Local shard", 19, anchor="middle")
    fig.line(385, 192, 414, 192, arrow=True)
    fig.rect(24, 484, 752, 52, TEAL)
    fig.text(
        400,
        518,
        "Global sample shape on both processes: (64, 16, 4)",
        22,
        color="white",
        anchor="middle",
    )
    fig.save("processes-and-devices.svg")


def phases():
    fig = Figure(
        1010,
        "Phase-specific layouts on eight devices",
        "Sampling divides 64 chains into eight groups of eight. Operator evaluation "
        "redistributes 1024 configurations into two halves, each repeated across four "
        "squared-charge operator partitions. Contributions are summed over operators.",
    )
    fig.text(32, 48, "The same devices, different work layouts", 29, weight="bold")
    fig.text(32, 93, "1. Sampling: 8 partitions of the chain axis", 24, weight="bold")
    for device in range(8):
        x = 28 + (device % 4) * 192
        y = 119 + (device // 4) * 92
        fig.rect(x, y, 168, 75, SEAFOAM)
        fig.text(x + 84, y + 29, f"Device {device}", 22, weight="bold", anchor="middle")
        fig.text(x + 84, y + 56, "8 chains × 16", 20, anchor="middle")  # noqa: RUF001
    fig.text(400, 326, "Every shard contains all 4 configuration labels", 21, anchor="middle")
    fig.line(400, 346, 400, 397, arrow=True)
    fig.text(32, 442, "2. Operator evaluation: 4 × 2 mesh", 24, weight="bold")  # noqa: RUF001
    fig.text(282, 481, "Samples 0–511", 22, anchor="middle")  # noqa: RUF001
    fig.text(612, 481, "Samples 512–1023", 22, anchor="middle")  # noqa: RUF001
    for term in range(4):
        y = 505 + term * 83
        fig.text(80, y + 43, f"Q{term}²", 24, weight="bold", anchor="middle")
        for sample_half in range(2):
            x = 134 + sample_half * 330
            fig.rect(x, y, 296, 64, PEACH if term % 2 == 0 else TERRA)
            fig.text(
                x + 148, y + 27, f"Operator partition {term}", 21, weight="bold", anchor="middle"
            )
            fig.text(x + 148, y + 51, f"Sample partition {sample_half}", 20, anchor="middle")
    fig.text(400, 868, "Sum the four term contributions for each sample", 22, anchor="middle")
    fig.line(400, 884, 400, 923, arrow=True)
    fig.rect(100, 940, 600, 44, TEAL)
    fig.text(
        400,
        970,
        "Reduce the global sample batch to an expectation",
        21,
        color="white",
        anchor="middle",
    )
    fig.save("phase-layouts.svg")


if __name__ == "__main__":
    sample_sharding()
    processes()
    phases()
