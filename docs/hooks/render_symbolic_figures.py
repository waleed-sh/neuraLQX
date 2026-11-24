"""Regenerate original SVG diagrams for the symbolic operator guides."""

# Mathematical symbols in figure labels are intentional.
# ruff: noqa: RUF001

from html import escape
from pathlib import Path

DESTINATION = Path(__file__).resolve().parents[1] / "assets/images/symbolic"
INK = "#18363b"
TEAL = "#006d77"
SEAFOAM = "#83c5be"
ICE = "#edf6f9"
PEACH = "#ffddd2"
TERRA = "#e29578"


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
            f'<g font-family="Arial, Helvetica, sans-serif" fill="{INK}">',
        ]
        self.rect(0, 0, 800, height, ICE, radius=18)
        self.text(30, 47, title, size=28, weight="bold")

    def rect(self, x, y, width, height, fill="white", stroke="none", radius=10):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

    def text(self, x, y, value, size=20, color=INK, weight="normal", anchor="start"):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'
        )

    def lines(self, x, y, values, size=20, spacing=29, **kwargs):
        for i, value in enumerate(values):
            self.text(x, y + spacing * i, value, size=size, **kwargs)

    def path(self, points, arrow=True):
        coordinates = " L".join(f"{x} {y}" for x, y in points)
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(
            f'<path d="M{coordinates}" fill="none" stroke="{TEAL}" '
            f'stroke-width="2.5" stroke-linejoin="round"{marker}/>'
        )

    def save(self, name):
        DESTINATION.mkdir(parents=True, exist_ok=True)
        (DESTINATION / name).write_text("\n".join([*self.parts, "</g></svg>\n"]))


def workflow():
    fig = Figure(
        490,
        "From a physical rule to sampled connections",
        "A Python symbolic definition becomes immutable IR and a native compiled operator. "
        "During a calculation, that operator accepts sampled configurations and returns "
        "connected configurations and matrix elements without building a dense matrix.",
    )
    fig.text(30, 85, "Define and compile once for this operator")
    for x, title, lines, fill in (
        (30, "Definition", ("Sites and conditions", "Updates and coefficients"), SEAFOAM),
        (290, "IR", ("Ordered terms", "Inspectable expressions"), "white"),
        (550, "Native operator", ("Static connection shape", "Generated JAX kernel"), PEACH),
    ):
        fig.rect(x, 113, 220, 133, fill, stroke=TEAL)
        fig.text(x + 110, 150, title, size=24, weight="bold", anchor="middle")
        fig.lines(x + 110, 188, lines, size=18, anchor="middle")
    fig.path(((255, 180), (283, 180)))
    fig.path(((515, 180), (543, 180)))
    fig.text(30, 296, "Reuse the operator throughout the calculation")
    fig.rect(30, 320, 740, 138, "white")
    fig.lines(145, 365, ("Sampled source", "σ with N labels"), size=21, anchor="middle")
    fig.rect(295, 340, 210, 89, TEAL)
    fig.lines(400, 374, ("Connection kernel", "A[σ, σ′]"), size=21, color="white", anchor="middle")
    fig.lines(656, 365, ("Targets σ′", "Matrix elements"), size=21, anchor="middle")
    fig.path(((245, 383), (283, 383)))
    fig.path(((516, 383), (557, 383)))
    fig.save("workflow.svg")


def emissions():
    fig = Figure(
        470,
        "Emissions add matrix elements at equal targets",
        "Four emissions start from source (1, 0). Two reach target (0, 1) with "
        "coefficients 2 and 3, giving 5. Two remain at (1, 0) with coefficients "
        "minus 1 and plus 1, which cancel. One active connection remains.",
    )
    fig.text(30, 85, "Each emission starts from the same source configuration")
    fig.rect(30, 209, 150, 95, SEAFOAM, stroke=TEAL)
    fig.lines(105, 246, ("Source", "(1, 0)"), size=24, anchor="middle")
    for i, label in enumerate(("(0, 1)     +2", "(0, 1)     +3", "(1, 0)     −1", "(1, 0)     +1")):
        y = 119 + i * 76
        fig.rect(280, y, 200, 55, PEACH if i < 2 else "white", stroke=TEAL)
        fig.text(380, y + 35, label, size=23, anchor="middle")
        fig.path(((180, 257), (222, 257), (222, y + 28), (271, y + 28)))
    fig.rect(595, 158, 175, 94, TEAL)
    fig.lines(682, 193, ("(0, 1)", "2 + 3 = 5"), size=23, color="white", anchor="middle")
    fig.rect(595, 305, 175, 94, "white", stroke=TERRA)
    fig.lines(682, 340, ("−1 + 1 = 0", "No active entry"), size=19, anchor="middle")
    for y in (147, 223):
        fig.path(((480, y), (531, y), (531, 204), (585, 204)))
    for y in (299, 375):
        fig.path(((480, y), (531, y), (531, 351), (585, 351)))
    fig.text(400, 441, "Four contributions, one nonzero connection", size=22, anchor="middle")
    fig.save("emissions.svg")


def pipeline():
    fig = Figure(
        780,
        "The symbolic compilation pipeline",
        "Validation, diagnostics, and normalisation precede cache lookup. A hit returns "
        "the stored artifact. A miss runs capacity, connectivity, cost, fusion, and kernel "
        "planning, then JAX lowering and native wrapping. Device compilation happens "
        "when the numerical function is called, separately from artifact caching.",
    )
    fig.rect(30, 90, 465, 62, SEAFOAM)
    fig.text(262, 129, "Definition, IR, and resolved target", size=23, anchor="middle")
    fig.path(((262, 155), (262, 184)))
    fig.rect(30, 195, 465, 112, "white", stroke=TEAL)
    fig.text(262, 228, "Before cache lookup", size=23, weight="bold", anchor="middle")
    fig.lines(
        262, 262, ("Validation and diagnostics", "Normalisation and signature"), anchor="middle"
    )
    fig.path(((262, 310), (262, 339)))
    fig.rect(30, 350, 465, 65, PEACH, stroke=TERRA)
    fig.text(262, 390, "Artifact cache lookup", size=24, anchor="middle")
    fig.path(((500, 383), (549, 383)))
    fig.text(525, 363, "Hit", size=18, anchor="middle")
    fig.rect(560, 343, 205, 106, TEAL)
    fig.lines(
        662, 380, ("Native operator", "Reports and plans"), size=21, color="white", anchor="middle"
    )
    fig.path(((262, 420), (262, 456)))
    fig.text(278, 446, "Miss", size=18)
    fig.rect(30, 467, 465, 154, "white", stroke=TEAL)
    fig.text(262, 501, "Analyse and plan", size=23, weight="bold", anchor="middle")
    fig.lines(
        262,
        538,
        (
            "Raw capacity and target equivalence",
            "Collision obligations and assembly cost",
            "Fusion and structured kernel plan",
        ),
        size=20,
        anchor="middle",
    )
    fig.path(((262, 625), (262, 654)))
    fig.rect(30, 665, 465, 65, SEAFOAM)
    fig.text(262, 705, "JAX lowering and native wrapping", size=23, anchor="middle")
    fig.path(((500, 698), (784, 698), (784, 397), (771, 397)))
    fig.path(((662, 455), (662, 498)))
    fig.rect(560, 509, 205, 112, PEACH)
    fig.lines(
        662, 541, ("Numerical call", "JAX tracing", "Device compilation"), size=20, anchor="middle"
    )
    fig.text(
        400,
        760,
        "Artifact reuse and device executable reuse are separate",
        size=20,
        anchor="middle",
    )
    fig.save("compiler-pipeline.svg")


def kernel_plan():
    fig = Figure(
        570,
        "Separate scalar work from target construction",
        "Instantiated symbolic actions supply guards and amplitudes to scalar batches, "
        "and ordered updates to coordinate maps. Target representatives are constructed "
        "once per equivalent class. Assembly adds colliding amplitudes and compacts "
        "the native padded arrays with separate batch, connection, and coordinate axes.",
    )
    fig.rect(30, 93, 740, 75, SEAFOAM, stroke=TEAL)
    fig.text(400, 125, "R instantiated actions", size=24, weight="bold", anchor="middle")
    fig.text(400, 153, "Term × iterator row × emission", size=20, anchor="middle")
    fig.path(((202, 172), (202, 218)))
    fig.path(((597, 172), (597, 218)))
    fig.rect(30, 229, 345, 136, "white", stroke=TEAL)
    fig.rect(425, 229, 345, 136, PEACH, stroke=TERRA)
    fig.text(202, 264, "Scalar expressions", size=24, weight="bold", anchor="middle")
    fig.lines(
        202, 302, ("Guards and amplitudes", "Batch matching templates"), size=21, anchor="middle"
    )
    fig.text(597, 264, "Target expressions", size=24, weight="bold", anchor="middle")
    fig.lines(
        597,
        302,
        ("Changed coordinates", "Representatives of equal targets"),
        size=20,
        anchor="middle",
    )
    fig.path(((202, 369), (202, 397), (300, 397), (300, 422)))
    fig.path(((597, 369), (597, 397), (500, 397), (500, 422)))
    fig.rect(170, 432, 460, 111, TEAL)
    fig.text(
        400, 467, "Assemble connections", size=24, color="white", weight="bold", anchor="middle"
    )
    fig.lines(
        400,
        501,
        ("Sum equal targets and compact", "Targets (..., K, N)    Elements (..., K)"),
        size=20,
        color="white",
        anchor="middle",
    )
    fig.save("kernel-plan.svg")


if __name__ == "__main__":
    workflow()
    emissions()
    pipeline()
    kernel_plan()
