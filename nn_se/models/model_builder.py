from .phasen import PHASEN
from .phasen import NetPHASEN
from ..FLAGS import PARAM

def get_model_class_and_var():
  model_class, var_class = {
      'PHASEN': (PHASEN, NetPHASEN),
  }[PARAM.model_name]

  return model_class, var_class
