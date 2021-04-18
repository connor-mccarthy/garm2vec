#!/usr/bin/env python

from cleo import Application
from garm2vec.console.plot_command import PlotCommand
from garm2vec.console.vector_command import VectorCommand

application = Application("garm2vec", "0.1.0")
application.add(VectorCommand())
application.add(PlotCommand())


def main() -> int:
    return application.run()


if __name__ == "__main__":
    main()
