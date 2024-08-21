"""An Architecture abstract class which allows descendants to load from configurations, and
ml modules to observe the module in a standardized way to do things such as hyperparameter
tuning on any given architecture
"""


from .ml_module_config import MLModuleConfig


class MLModule():

    def from_config(self, config: MLModuleConfig):
        return self.__init__(
            *config.unnamed_modules,
            **config.named_modules
        )