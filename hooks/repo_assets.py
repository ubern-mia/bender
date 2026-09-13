"""Publish figures that live next to the code (outside ``docs/``) to the built site.

Episode figures are committed alongside the scripts that produced them, e.g.
``training-models/dermamnist_v7_with_augm/test_accuracy_v7.png``. Copying them into
``docs/`` would duplicate them in git and leave two copies to keep in sync, so instead this
hook registers each one as a virtual MkDocs file under ``figures/<original path>``.

Episode pages then reference them with an ordinary relative link::

    ![...](../figures/training-models/dermamnist_v7_with_augm/test_accuracy_v7.png)

which also means MkDocs' own link validation covers them. Works for both ``mkdocs build``
and ``mkdocs serve``.
"""

from pathlib import Path

from mkdocs.structure.files import File

#: Directories, relative to the repository root, whose images should be published.
SOURCE_DIRS = (
    "training-models",
    "federated-learning",
    "exploratory-data-analysis",
)

SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg"}

#: Prefix under which the figures are served.
URL_PREFIX = "figures"


def on_files(files, config):
    root = Path(config.config_file_path).parent

    for source in SOURCE_DIRS:
        base = root / source
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in SUFFIXES or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            files.append(
                File.generated(
                    config,
                    f"{URL_PREFIX}/{relative}",
                    abs_src_path=str(path),
                )
            )

    return files
