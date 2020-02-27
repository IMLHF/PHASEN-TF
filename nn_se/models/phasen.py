import tensorflow as tf
import collections

from .modules import Module
from .modules import FrowardOutputs
from .modules import Losses
from ..FLAGS import PARAM
from ..utils import losses
from ..utils import misc_utils


class Stream_PreNet:
  def __init__(self, channel_out, kernels=[[1,7], [7,1]], name='streamA_or_P_prenet'):
    '''
    channel_out: output channel
    kernels: kernel for layers
    '''
    self.conv2d_lst = []
    for i, kernel in enumerate(kernels):
      conv_name = name+("_%d" % i)
      conv2d = tf.keras.layers.Conv2D(filters=channel_out, kernel_size=kernel, padding="same",name=conv_name)
      self.conv2d_lst.append(conv2d)

  def __call__(self, feature_in):
    '''
    feature_in : [batch, T, F, channel_in]
    return : [batch, T, F, channel_out]
    '''
    if len(self.conv2d_lst) == 0:
      return feature_in
    out = feature_in
    for conv2d in self.conv2d_lst:
      out = conv2d(out)
    return out


class BatchNormAndActivate:
  def __init__(self, bn_axis=-1, activation=tf.keras.activations.relu, name='BN_activate'):
    self.bn_layer = tf.keras.layers.BatchNormalization(bn_axis, name=name)
    self.activate_fn = activation

  def __call__(self, fea_in):
    return self.activate_fn(self.bn_layer(fea_in))


class FrequencyTransformationBlock:
  def __init__(self, frequency_dim, channel_in_out, channel_attention=5, name="FTB"):
    self.att_conv2d_1 = tf.keras.layers.Conv2D(channel_attention, [1,1], padding="same",
                                               name=name+"/att_conv2d_1")  # [batch, T, F * channel_attention]
    self.att_conv2d_1_bna = BatchNormAndActivate(name=name+"/att_conv2d_1_bna")

    self.att_inner_reshape = tf.keras.layers.Reshape([-1, frequency_dim * channel_attention])
    self.att_conv1d_2 = tf.keras.layers.Conv1D(frequency_dim, 9, padding="same",
                                               name=name+"/att_conv1d_2")  # [batch, T, F]
    self.att_conv1d_2_bna = BatchNormAndActivate(name=name+"/att_conv1d_2_bna")
    self.att_out_reshape = tf.keras.layers.Reshape([-1, frequency_dim, 1])
    self.frequencyFC = tf.keras.layers.Dense(frequency_dim, name=name+"/FFC")
    self.concat_FFCout_and_In = tf.keras.layers.Concatenate(-1)

    self.out_conv2d = tf.keras.layers.Conv2D(channel_in_out, [1,1], padding="same",
                                             name=name+"/out_conv2d")
    self.out_conv2d_bna = BatchNormAndActivate(name=name+"/out_conv2d_bna")

  def __call__(self, feature_in):
    '''
    feature_n: [batch, T, F, channel_in_out]
    '''
    att_out = self.att_conv2d_1(feature_in)
    att_out = self.att_conv2d_1_bna(att_out)

    att_out = self.att_inner_reshape(att_out)
    att_out = self.att_conv1d_2(att_out)
    att_out = self.att_conv1d_2_bna(att_out)
    att_out = self.att_out_reshape(att_out)

    atted_out = tf.multiply(feature_in, att_out) # [batch, T, F, channel_in_out]
    atted_out_T = tf.transpose(atted_out, perm=[0,1,3,2]) # [batch, T, channel_in_out, F]
    ffc_out_T = self.frequencyFC(atted_out_T)
    ffc_out = tf.transpose(ffc_out_T, perm=[0,1,3,2]) # [batch, T, F, channel_in_out]
    concated_out = self.concat_FFCout_and_In([feature_in, ffc_out])

    out = self.out_conv2d(concated_out)
    out = self.out_conv2d_bna(out)
    return out


class InfoCommunicate:
  def __init__(self, channel_out, activate_fn=tf.keras.activations.tanh, name='InfoC'):
    self.conv2d = tf.keras.layers.Conv2D(channel_out, [1, 1], padding="same", name=name+"/conv2d")
    # self.bn = tf.keras.layers.BatchNormalization(-1, name=name+"/BN")
    self.activate_fn = activate_fn

  def __call__(self, feature_x1, feature_x2):
    # feature_x1: [batch, T, F, channel_out]
    # feature_x2: [batch, T, F, Cp or Ca]
    # return: [batch, T, F, channel_out]
    conv_out = self.conv2d(feature_x2)
    act_out = self.activate_fn(conv_out)

    out_multiply = tf.multiply(feature_x1, act_out)
    return out_multiply


class TwoStreamBlock:
  def __init__(self, frequency_dim, channel_in_out_A, channel_in_out_P, name="TSB"):
    self.sA1_pre_FTB = FrequencyTransformationBlock(frequency_dim, channel_in_out_A, name=name+"/sA1_pre_FTB")
    self.sA2_conv2d = tf.keras.layers.Conv2D(channel_in_out_A, [5, 5], padding="same", name=name+"/sA2_conv2d")
    self.sA2_conv2d_bna = BatchNormAndActivate(name=name+"/sA2_conv2d_bna")
    self.sA3_conv2d = tf.keras.layers.Conv2D(channel_in_out_A, [25, 1], padding="same", name=name+"/sA3_conv2d")
    self.sA3_conv2d_bna = BatchNormAndActivate(name=name+"/sA3_conv2d_bna")
    self.sA4_conv2d = tf.keras.layers.Conv2D(channel_in_out_A, [5, 5], padding="same", name=name+"/sA4_conv2d")
    self.sA4_conv2d_bna = BatchNormAndActivate(name=name+"/sA4_conv2d_bna")
    self.sA5_post_FTB = FrequencyTransformationBlock(frequency_dim, channel_in_out_A, name=name+"/sA5_post_FTB")
    self.sA6_info_communicate = InfoCommunicate(channel_in_out_A, name=name+"/InfoC_A")

    self.sP1_conv2d_before_LN = tf.keras.layers.LayerNormalization(-1, name=name+"/sP1_conv2d_before_LN")
    self.sP1_conv2d = tf.keras.layers.Conv2D(channel_in_out_P, [5, 3], padding="same", name=name+"/sP1_conv2d")
    self.sP2_conv2d_before_LN = tf.keras.layers.LayerNormalization(-1, name=name+"/sP2_conv2d_before_LN")
    self.sP2_conv2d = tf.keras.layers.Conv2D(channel_in_out_P, [25, 1], padding="same", name=name+"/sP2_conv2d")
    self.sP3_info_communicate = InfoCommunicate(channel_in_out_P, name=name+"/InfoC_P")

  def __call__(self, feature_sA, feature_sP):
    # Stream A
    sA_out = feature_sA
    sA_out = self.sA1_pre_FTB(sA_out)
    sA_out = self.sA2_conv2d(sA_out)
    sA_out = self.sA2_conv2d_bna(sA_out)
    sA_out = self.sA3_conv2d(sA_out)
    sA_out = self.sA3_conv2d_bna(sA_out)
    sA_out = self.sA4_conv2d(sA_out)
    sA_out = self.sA4_conv2d_bna(sA_out)
    sA_out = self.sA5_post_FTB(sA_out)

    # Strean P
    sP_out = feature_sP
    sP_out = self.sP1_conv2d_before_LN(sP_out)
    sP_out = self.sP1_conv2d(sP_out)
    sP_out = self.sP2_conv2d_before_LN(sP_out)
    sP_out = self.sP2_conv2d(sP_out)

    # information communication
    sA_fin_out = self.sA6_info_communicate(sA_out, sP_out)
    sP_fin_out = self.sP3_info_communicate(sP_out, sA_out)

    return sA_fin_out, sP_fin_out


class StreamAmplitude_PostNet:
  def __init__(self, frequency_dim, name="sA_PostNet"):
    self.layer_sequences = []

    self.layer_sequences.append(
        tf.keras.layers.Conv2D(8, [1, 1], padding="same", name=name+"/p1_conv2d"))
    self.layer_sequences.append(tf.keras.layers.Reshape([-1, frequency_dim * 8]))

    fw_lstm = tf.keras.layers.LSTM(512, dropout=0.2, implementation=2,
                                   return_sequences=True, name='fwlstm')
    bw_lstm = tf.keras.layers.LSTM(512, dropout=0.2, implementation=2,
                                   return_sequences=True, name='bwlstm', go_backwards=True)
    self.layer_sequences.append(
        tf.keras.layers.Bidirectional(layer=fw_lstm, backward_layer=bw_lstm,
                                      merge_mode='concat', name=name+'/p2_blstm'))

    self.layer_sequences.append(
        tf.keras.layers.Dense(600, activation=tf.keras.activations.relu, name=name+"/p3_dense"))
    self.layer_sequences.append(
        tf.keras.layers.Dense(600, activation=tf.keras.activations.relu, name=name+"/p4_dense"))
    self.layer_sequences.append(
        tf.keras.layers.Dense(frequency_dim, activation=tf.keras.activations.sigmoid,
                              name=name+"/out_dense"))

  def __call__(self, feature_sA):
    '''
    return [batch, T, F]
    '''
    out = feature_sA
    for layer_fn in self.layer_sequences:
      out = layer_fn(out)
    return out


class StreamPhase_PostNet:
  def __init__(self, name="sP_PostNet"):
    self.layer_sequences = []
    self.layer_sequences.append(
      tf.layers.Conv2D(2, [1,1], padding="same", name=name+"/conv2d"))

  def __call__(self, feature_sP):
    '''
    return [batch, T, F]->complex
    '''
    out = feature_sP
    for layer_fn in self.layer_sequences:
      out = layer_fn(out)
    # out: [batch, T, F, 2]
    out_complex = tf.complex(out[..., 0], out[..., 1])
    out_angle = tf.angle(out_complex)
    normed_out = tf.exp(tf.complex(0.0, out_angle))
    return normed_out


class NET_PHASEN_OUT(
    collections.namedtuple("NET_PHASEN_OUT",
                           ("mag", "normalized_complex_phase"))):
  pass

def if_grads_is_nan_or_inf(grads):
  grads_is_nan_or_inf_lst = []
  for grad in grads:
    nan_or_inf = tf.reduce_max(tf.cast(tf.math.is_nan(grad), tf.int32)) + \
        tf.reduce_max(tf.cast(tf.math.is_inf(grad), tf.int32))
    grads_is_nan_or_inf_lst.append(nan_or_inf)
  grads_is_nan_or_inf = tf.stack(grads_is_nan_or_inf_lst)
  # True if grads_is_nan_or_inf > 0 else False
  return grads_is_nan_or_inf

class PHASEN(Module):
  def __init__(self,
               mode,
               mixed_wav_batch,
               clean_wav_batch=None,
               noise_wav_batch=None):
    super(PHASEN, self).__init__(
        mode,
        mixed_wav_batch,
        clean_wav_batch,
        noise_wav_batch)

    # get specific variables
    self.var_scope = None
    self.se_net_vars = tf.compat.v1.trainable_variables(self.var_scope)

    # show specific variables
    if mode == PARAM.MODEL_TRAIN_KEY:
      print("\nSE PARAMs")
      misc_utils.show_variables(self.se_net_vars)

    if mode == PARAM.MODEL_VALIDATE_KEY or mode == PARAM.MODEL_INFER_KEY:
      return

    # region get specific grads
    ## se_net grads
    se_loss_grads = tf.gradients(
      self._losses.sum_loss,
      self.se_net_vars,
      colocate_gradients_with_ops=True
    )
    # endregion

    all_grads = se_loss_grads
    all_params = self.se_net_vars

    # all_clipped_grads, _ = tf.clip_by_global_norm(all_grads, PARAM.max_gradient_norm)
    all_clipped_grads = all_grads


    # choose optimizer
    if PARAM.optimizer == "Adam":
      self._optimizer = tf.compat.v1.train.AdamOptimizer(self._lr)
    elif PARAM.optimizer == "RMSProp":
      self._optimizer = tf.compat.v1.train.RMSPropOptimizer(self._lr)

    self._grads_bad_lst = if_grads_is_nan_or_inf(all_clipped_grads)
    self._grads_bad = tf.greater(tf.reduce_max(self._grads_bad_lst), 0)
    self._grads_coef = tf.cond(self._grads_bad,
                               lambda: tf.constant(0.0),
                               lambda: tf.constant(1.0))
    checked_grads = [tf.math.multiply_no_nan(grad, self._grads_coef) for grad in all_clipped_grads]
    self._train_op = self._optimizer.apply_gradients(zip(checked_grads, all_params),
                                                     global_step=self.global_step)

  def _init_variables(self):

    self.streamA_prenet = Stream_PreNet(PARAM.channel_A, PARAM.prenet_A_kernels, name='streamA_prenet')
    self.streamP_prenet = Stream_PreNet(PARAM.channel_P, PARAM.prenet_P_kernels, name='streanP_prenet')
    self.layers_TSB = []
    for i in range(1, PARAM.n_TSB+1):
      tsb_t = TwoStreamBlock(PARAM.frequency_dim, PARAM.channel_A, PARAM.channel_P, name="TSB_%d" % i)
      self.layers_TSB.append(tsb_t)
    self.streamA_postnet = StreamAmplitude_PostNet(PARAM.frequency_dim, name="sA_postnet")
    self.streamP_postnet = StreamPhase_PostNet(name="sP_postnet")

  def _net_PHASEN(self, feature_in):
    '''
    return mag_batch[batch, time, fre]->real, normalized_complex_phase[batch, time, fre]->complex
    '''
    sA_out = self.streamA_prenet(feature_in) # [batch, t, f, Ca]
    sP_out = self.streamP_prenet(feature_in) # [batch, t, f, Cp]
    for tsb in self.layers_TSB:
      sA_out, sP_out = tsb(sA_out, sP_out)
    sA_out = self.streamA_postnet(sA_out) # [batch, t, f]
    sP_out = self.streamP_postnet(sP_out) # [batch, t, f, 2]

    est_mag = tf.multiply(self.mixed_mag_batch, sA_out) # [batch, t, f]
    normed_complex_phase = sP_out # [batch, t, f, 2]
    return NET_PHASEN_OUT(mag=est_mag,
                          normalized_complex_phase=normed_complex_phase)

  def _forward(self):
    mixed_stft_batch_real = tf.real(self.mixed_stft_batch)
    mixed_stft_batch_imag = tf.imag(self.mixed_stft_batch)
    mixed_stft_batch_real = tf.expand_dims(mixed_stft_batch_real, -1)
    mixed_stft_batch_imag = tf.expand_dims(mixed_stft_batch_imag, -1)
    feature_in = tf.concat([mixed_stft_batch_real, mixed_stft_batch_imag], axis=-1)

    net_phasen_out = self._net_PHASEN(feature_in)

    est_clean_mag_batch = net_phasen_out.mag
    est_complexPhase_batch = net_phasen_out.normalized_complex_phase
    est_clean_stft_batch = tf.multiply(tf.complex(est_clean_mag_batch, 0.0), est_complexPhase_batch)
    est_clean_wav_batch = misc_utils.tf_stft2wav(est_clean_stft_batch, PARAM.frame_length,
                                                 PARAM.frame_step, PARAM.fft_length)
    _mixed_wav_length = tf.shape(self.mixed_wav_batch)[1]
    est_clean_wav_batch = est_clean_wav_batch[:, :_mixed_wav_length]

    return FrowardOutputs(est_clean_stft_batch,
                          est_clean_mag_batch,
                          est_clean_wav_batch,
                          est_complexPhase_batch)


  def _get_losses(self):
    est_clean_mag_batch = self._forward_outputs.est_clean_mag_batch
    est_clean_stft_batch = self._forward_outputs.est_clean_stft_batch
    est_clean_wav_batch = self._forward_outputs.est_clean_wav_batch
    # est_complexPhase_batch = self._forward_outputs.est_complexPhase_batch

    # region losses
    self.loss_compressedMag_mse = losses.batch_time_compressedMag_mse(est_clean_mag_batch,
                                                                      self.clean_mag_batch,
                                                                      PARAM.loss_compressedMag_idx)
    self.loss_compressedStft_mse = losses.batch_time_compressedStft_mse(est_clean_stft_batch,
                                                                        self.clean_stft_batch,
                                                                        PARAM.loss_compressedMag_idx)
    self.loss_mag_mse = losses.batch_time_fea_real_mse(est_clean_mag_batch, self.clean_mag_batch)
    self.loss_mag_reMse = losses.batch_real_relativeMSE(est_clean_mag_batch, self.clean_mag_batch,
                                                        PARAM.relative_loss_epsilon, PARAM.RL_idx)
    self.loss_stft_mse = losses.batch_time_fea_complex_mse(est_clean_stft_batch, self.clean_stft_batch)
    self.loss_stft_reMse = losses.batch_complex_relativeMSE(est_clean_stft_batch, self.clean_stft_batch,
                                                            PARAM.relative_loss_epsilon, PARAM.RL_idx)

    self.loss_wav_L1 = losses.batch_wav_L1_loss(est_clean_wav_batch, self.clean_wav_batch)*10.0
    self.loss_wav_L2 = losses.batch_wav_L2_loss(est_clean_wav_batch, self.clean_wav_batch)*100.0
    self.loss_wav_reL2 = losses.batch_wav_relativeMSE(est_clean_wav_batch, self.clean_wav_batch,
                                                      PARAM.relative_loss_epsilon, PARAM.RL_idx)

    self.loss_CosSim = losses.batch_CosSim_loss(est_clean_wav_batch, self.clean_wav_batch)
    self.loss_SquareCosSim = losses.batch_SquareCosSim_loss(est_clean_wav_batch, self.clean_wav_batch)
    self.loss_stCosSim = losses.batch_short_time_CosSim_loss(est_clean_wav_batch, self.clean_wav_batch,
                                                             PARAM.st_frame_length_for_loss,
                                                             PARAM.st_frame_step_for_loss)
    self.loss_stSquareCosSim = losses.batch_short_time_SquareCosSim_loss(est_clean_wav_batch, self.clean_wav_batch,
                                                                         PARAM.st_frame_length_for_loss,
                                                                         PARAM.st_frame_step_for_loss)
    loss_dict = {
        'loss_compressedMag_mse': self.loss_compressedMag_mse,
        'loss_compressedStft_mse': self.loss_compressedStft_mse,
        'loss_mag_mse': self.loss_mag_mse,
        'loss_mag_reMse': self.loss_mag_reMse,
        'loss_stft_mse': self.loss_stft_mse,
        'loss_stft_reMse': self.loss_stft_reMse,
        'loss_wav_L1': self.loss_wav_L1,
        'loss_wav_L2': self.loss_wav_L2,
        'loss_wav_reL2': self.loss_wav_reL2,
        'loss_CosSim': self.loss_CosSim,
        'loss_SquareCosSim': self.loss_SquareCosSim,
        'loss_stCosSim': self.loss_stCosSim,
        'loss_stSquareCosSim': self.loss_stSquareCosSim,
    }
    # endregion losses

    # region sum_loss
    sum_loss = tf.constant(0, dtype=tf.float32)
    sum_loss_names = PARAM.sum_losses
    for i, name in enumerate(sum_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.sum_losses_w) > 0:
        loss_t *= PARAM.sum_losses_w[i]
      sum_loss += loss_t
    # endregion sum_loss

    # region show_losses
    show_losses = []
    show_loss_names = PARAM.show_losses
    for i, name in enumerate(show_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.show_losses_w) > 0:
        loss_t *= PARAM.show_losses_w[i]
      show_losses.append(loss_t)
    show_losses = tf.stack(show_losses)
    # endregion show_losses

    # region stop_criterion_losses
    stop_criterion_losses_sum = tf.constant(0, dtype=tf.float32)
    stop_criterion_loss_names = PARAM.stop_criterion_losses
    for i, name in enumerate(stop_criterion_loss_names):
      loss_t = loss_dict[name]
      if len(PARAM.stop_criterion_losses_w) > 0:
        loss_t *= PARAM.stop_criterion_losses_w[i]
      stop_criterion_losses_sum += loss_t
    # endregion stop_criterion_losses

    return Losses(sum_loss=sum_loss,
                  show_losses=show_losses,
                  stop_criterion_loss=stop_criterion_losses_sum)
