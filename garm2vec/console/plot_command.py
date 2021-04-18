import os

from bashplotlib.scatterplot import plot_scatter
from cleo import Command
from cleo.helpers import argument
from garm2vec import Garm2Vec
from garm2vec.constants import EMBEDDING_DIMS


class PlotCommand(Command):
    name = "plot"
    arguments = [
        argument(
            "filepath",
            "Path to image.",
        ),
        argument(
            "description",
            "Description text.",
        ),
    ]
    f"Pass a filepath and description argument to generate a plot of the {EMBEDDING_DIMS} dimensional garment embedding."

    def handle(self) -> None:
        g2v = Garm2Vec()
        filepath = self.argument("filepath")
        full_filepath = os.path.abspath(os.path.expanduser(filepath))
        description = self.argument("description")
        vector = g2v.get_one([full_filepath, description])
        data = [f"{x},{y}" for x, y in enumerate(vector)]
        plot_scatter(
            f=data,
            xs=None,
            ys=None,
            title="Garm2Vec",
            pch="o",
            size=40,
            colour="default",
        )
