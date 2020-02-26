from .phasen import PHASEN
from . import modules
from ..FLAGS import PARAM

def get_model_class_and_var():
  model_class = {
      'PHASEN': PHASEN,
  }[PARAM.model_name]

  return model_class
