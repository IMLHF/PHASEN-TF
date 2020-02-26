import tensorflow as tf
import abc
import collections
from typing import Union

from ..FLAGS import PARAM
from ..utils import misc_utils

class FrowardOutputs(
    collections.namedtuple("FrowardOutputs",
                           ("est_clean_stft_batch", "est_clean_mag_batch",
                            "est_clean_wav_batch"))):
  pass

class Losses(
    collections.namedtuple("Losses",
                           ("sum_loss", "show_losses", "stop_criterion_loss"))):
  pass


class Module(object):
  """
  speech enhancement base.
  Discriminate spec and mag:
    spec: spectrum, complex value.
    mag: magnitude, real value.
  """
  def __init__(self,
               mode,
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    del noise_wav_batch
    self.mode = mode
    self.__var_se_net_scope = 'se_net/'
    self._init_variables()

    self.mixed_wav_batch = mixed_wav_batch
    self.mixed_stft_batch = misc_utils.tf_wav2stft(self.mixed_wav_batch,
                                                   PARAM.frame_length,
                                                   PARAM.frame_step)
    self.mixed_mag_batch = tf.abs(self.mixed_stft_batch)
    self.mixed_angle_batch = tf.angle(self.mixed_stft_batch)

    if clean_wav_batch is not None:
      self.clean_wav_batch = clean_wav_batch
      self.clean_stft_batch = misc_utils.tf_wav2stft(self.clean_wav_batch,
                                                     PARAM.frame_length,
                                                     PARAM.frame_step)
      self.clean_mag_batch = tf.abs(self.clean_stft_batch)
      self.clean_angle_batch = tf.angle(self.clean_stft_batch)


    # global_step, lr, vars
    with tf.compat.v1.variable_scope("notrain_var", reuse=tf.compat.v1.AUTO_REUSE):
      self._global_step = tf.compat.v1.get_variable('global_step', dtype=tf.int32,
                                                    initializer=tf.constant(1), trainable=False)
      self._lr = tf.compat.v1.get_variable('lr', dtype=tf.float32, trainable=False,
                                           initializer=tf.constant(PARAM.learning_rate))
    self.save_variables = [self.global_step, self._lr]

    # for lr halving
    self._new_lr = tf.compat.v1.placeholder(tf.float32, name='new_lr')
    self._assign_lr = tf.compat.v1.assign(self._lr, self.new_lr)

    # for lr warmup
    if PARAM.use_lr_warmup:
      self._lr = misc_utils.noam_scheme(self._lr, self.global_step, warmup_steps=PARAM.warmup_steps)

    # nn forward
    self._forward_outputs = self._forward()
    self._est_clean_wav_batch = self._forward_outputs.est_clean_wav_batch


    # get loss
    if mode != PARAM.MODEL_INFER_KEY:
      # losses
      self._losses = self._get_losses()
      self._sum_loss = self._losses.sum_loss

    # trainable_variables = tf.compat.v1.trainable_variables()
    self.se_net_params = tf.compat.v1.trainable_variables(scope=self.__var_se_net_scope)

    if mode == PARAM.MODEL_TRAIN_KEY:
      print("\nSE PARAMs")
      misc_utils.show_variables(self.se_net_params)

    self.save_variables.extend(self.se_net_params)
    self.saver = tf.compat.v1.train.Saver(self.save_variables,
                                          max_to_keep=PARAM.max_keep_ckpt,
                                          save_relative_paths=True)


  @abc.abstractmethod
  def _init_variables(self):
    '''
    create variables
    '''
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "_init_variables not implement, code: varkdiw93kk")


  @abc.abstractmethod
  def _forward(self):
    """
    Returns:
      forward_outputs: use for get_losses
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "_forward not implement, code: irjg0299gjjr")
    return FrowardOutputs()


  @abc.abstractmethod
  def _get_losses(self):
    """
    Returns:
      losses list
    """
    import traceback
    traceback.print_exc()
    raise NotImplementedError(
        "_get_losses not implement, code: 83jmvlwlviif")
    return Losses()

  def change_lr(self, sess, new_lr):
    sess.run(self._assign_lr, feed_dict={self.new_lr:new_lr})

  @property
  def global_step(self):
    return self._global_step

  @property
  def mixed_wav_batch_in(self):
    return self.mixed_wav_batch

  @property
  def train_op(self):
    return self._train_op

  @property
  def losses(self):
    return self._losses

  @property
  def lr(self):
    return self._lr

  @property
  def est_clean_wav_batch(self):
    return self._est_clean_wav_batch
