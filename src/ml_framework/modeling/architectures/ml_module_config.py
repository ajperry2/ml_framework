"""An object which loads yaml/json into an object which
MLModules can be loaded from
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import jsonschema


_module_schema = {
    # The parameters the module received
    # Instanctiated as module(*unnamed_modules, **named_modules)
    "unnamed_parameters": Optional[List[Any]],
    "named_parameters": Optional[Dict]
}


class MLModuleConfig:

    def load_yaml(self, path: Path, validate=True):
        data = yaml.safe_load(open(path, "r"))
        if validate:
            jsonschema.validate(data, _module_schema)
        self.named_parameters = data.get("named_parameters", dict())
        self.unnamed_parameters = data.get("unnamed_parameters", [])

    def load_json(self, path: Path, validate=True):
        data = json.load(open(path, "r"))
        if validate:
            jsonschema.validate(data, _module_schema)
        self.named_parameters = data.get("named_parameters", dict())
        self.unnamed_parameters = data.get("unnamed_parameters", [])
