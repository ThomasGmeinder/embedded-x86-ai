"""Run a notebook the way `jupyter nbconvert --execute` does, and also keep every redraw the kernel sends.

A cell that shows something and changes it later (`display(..., display_id=True)`, then `.update(...)`) leaves
only the last state in the executed notebook. A person at the notebook sees every state. This program keeps them:
it runs the notebook with the class `jupyter nbconvert --execute` runs it with (nbconvert's ExecutePreprocessor:
cells tagged "skip-execution" are left out, the first error ends the run) and writes, beside the executed
notebook, every "update_display_data" message of the kernel. That is the message on which a notebook front end
draws an output again that it already shows.

    python notebook_display_updates.py KERNEL NOTEBOOK EXECUTED.ipynb REDRAWS.json

Run it with a Python that has nbconvert (the workshop's .venv), in the directory the kernel is to start in.
REDRAWS.json is a list, in the order sent, of {"cell": position of the cell in the notebook, "data": what the
output shows from then on}.
"""
import json
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class KeepsRedraws(ExecutePreprocessor):
    """nbconvert's executor, unchanged in what it runs; it also keeps the redraw messages it is sent."""

    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "update_display_data":
            self.redraws.append({"cell": cell_index, "data": msg["content"]["data"]})
        return super().process_message(msg, cell, cell_index)


def main(kernel: str, source: str, executed: str, redraws: str) -> None:
    notebook = nbformat.read(source, as_version=4)
    executor = KeepsRedraws(kernel_name=kernel)
    executor.redraws = []
    executor.preprocess(notebook, {})  # no path given: the kernel starts in this program's directory
    nbformat.write(notebook, executed)
    with open(redraws, "w", encoding="utf-8") as fh:
        json.dump(executor.redraws, fh)


if __name__ == "__main__":
    main(*sys.argv[1:])
