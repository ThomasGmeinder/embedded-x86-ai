"""A Jupyter Server extension that says when the notebooks' server listens.

Jupyter Server logs its address ("is running at") before it listens: it prints
the address in ``start_app`` and begins to listen only inside its event loop,
which starts after that. A program that connects on the address line can be
refused. The server gives extensions one hook that runs inside the running
loop, after the callback that begins to listen (queued when the server is set
up, before this hook is): this extension logs one line from it.

``notebooks/run_notebooks.sh`` enables it; ``tests/test_notebooks_launcher.py``
waits for the line.
"""

from __future__ import annotations

LISTENING = "The notebooks' server listens on port"


def _jupyter_server_extension_points() -> list[dict]:
    return [{"module": __name__}]


def _load_jupyter_server_extension(serverapp) -> None:
    """Nothing to add to the server: no handler, no setting."""


async def _start_jupyter_server_extension(serverapp) -> None:
    serverapp.log.info("%s %d", LISTENING, serverapp.port)
