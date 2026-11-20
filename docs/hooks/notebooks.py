"""Expose downloads of the saved notebook sources copied by mkdocs-jupyter."""

from pathlib import PurePosixPath
from urllib.parse import quote


def on_page_content(html, *, page, **kwargs):
    if not page.file.src_uri.endswith(".ipynb"):
        return html
    filename = quote(PurePosixPath(page.file.src_uri).name)
    return f'<p class="nqx-notebook-download"><a href="{filename}" download>Download notebook</a></p>\n{html}'
