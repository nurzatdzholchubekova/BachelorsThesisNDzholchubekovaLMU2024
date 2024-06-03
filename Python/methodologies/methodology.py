from config import MethodologyConfiguration


class Methodology:
    def __init__(self, name: str, config: MethodologyConfiguration):
        self.name = name
        self.config = config
