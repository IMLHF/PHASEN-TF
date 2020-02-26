import tensorflow as tf
import collections
import os
from pathlib import Path

from ..FLAGS import PARAM
from ..utils import misc_utils
from ..utils import audio


class DataSetsOutputs(
    collections.namedtuple("DataSetOutputs",
                           ("initializer", "clean", "mixed"))):
  pass


def _generator(noisy_list, clean_list):
  """
  return clean, noisy
  """
  for noisy_dir, clean_dir in zip(noisy_list, clean_list):
    noisy, nsr = audio.read_audio(noisy_dir)
    clean, csr = audio.read_audio(clean_dir)
    assert nsr == csr, "sample rate error."
    wav_len = int(PARAM.train_val_wav_seconds*PARAM.sampling_rate)
    # clean = audio.repeat_to_len(clean, wav_len)
    # noisy = audio.repeat_to_len(noisy, wav_len)
    clean, noisy = audio.repeat_to_len_2(clean, noisy, wav_len, True)
    yield clean, noisy


def get_batch_inputs_from_nosiyCleanDataset(noisy_path, clean_path, shuffle_records=True):
  """
  noisy_path: noisy wavs path
  clean_path: clean wavs path
  """
  noisy_path = Path(noisy_path)
  clean_path = Path(clean_path)
  noisy_list = list(map(str, noisy_path.glob("*.wav")))
  clean_list = list(map(str, clean_path.glob("*.wav")))
  noisy_list.sort()
  clean_list.sort()

  wav_len = int(PARAM.train_val_wav_seconds*PARAM.sampling_rate)
  dataset = tf.data.Dataset.from_generator(_generator, output_types=(tf.float32,tf.float32),
                                           output_shapes=(
                                               tf.TensorShape([wav_len]),
                                               tf.TensorShape([wav_len])),
                                           args=(noisy_list, clean_list))

  if shuffle_records:
    dataset = dataset.shuffle(PARAM.batch_size*10)

  dataset = dataset.batch(batch_size=PARAM.batch_size, drop_remainder=True)
  # dataset = dataset.prefetch(buffer_size=PARAM.batch_size)
  dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
  clean, mixed = dataset_iter.get_next()
  return DataSetsOutputs(initializer=dataset_iter.initializer,
                         clean=clean,
                         mixed=mixed)
