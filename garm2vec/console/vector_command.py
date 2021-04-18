import os

from cleo import Command
from cleo.helpers import argument
from garm2vec import Garm2Vec
from garm2vec.constants import EMBEDDING_DIMS


class VectorCommand(Command):
    name = "vector"
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
    f"Pass a filepath and description argument to generate a {EMBEDDING_DIMS} dimensional garment embedding."

    def handle(self) -> None:
        g2v = Garm2Vec()
        filepath = self.argument("filepath")
        full_filepath = os.path.abspath(os.path.expanduser(filepath))
        description = self.argument("description")
        vector = g2v.get_one([full_filepath, description])
        self.line(str(vector))
