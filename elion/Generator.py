# elion/Generator.py  (factory only)
import importlib

class Generator:
    """
    Wraps a concrete generator class.

    Example
    -------
    gen = Generator(config).generator
    """

    def __init__(self, generator_cfg):
        name = generator_cfg["name"]          # e.g. "ReLeaSE"
        # module path inside the flat project
        module = importlib.import_module(f"elion.generators.{name}")
        cls = getattr(module, name)
        self.generator = cls(generator_cfg)
