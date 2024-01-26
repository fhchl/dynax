from equinox import Module

from dynax.util import pretty


class PrettyModule(Module):
    def pprint(self):
        """Print module with array values and parameter bounds."""
        print(pretty(self))
