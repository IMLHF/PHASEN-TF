from .phasen import PHASEN
from .phasen import PHASEN_Variables
from ..FLAGS import PARAM

def get_model_class_and_var():
  model_class, var_class = {
      'PHASEN': (PHASEN, PHASEN_Variables),
  }[PARAM.model_name]

  return model_class, var_class
