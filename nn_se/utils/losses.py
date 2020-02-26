import tensorflow as tf

def vec_dot_mul(y1, y2):
  dot_mul = tf.reduce_sum(tf.multiply(y1, y2), -1)
  return dot_mul

def vec_normal(y):
  normal_ = tf.sqrt(tf.reduce_sum(tf.square(y), -1))
  return normal_

def batch_time_compressedMag_mse(y1, y2, compress_idx):
  """
  y1>=0: real, [batch, time, feature_dim]
  y2>=0: real, [batch, time, feature_dim]
  """
  y1 = tf.pow(y1, compress_idx)
  y2 = tf.pow(y2, compress_idx)
  loss = tf.square(y1-y2)
  loss = tf.reduce_mean(tf.reduce_sum(loss, 0))
  return loss

def batch_time_fea_real_mse(y1, y2):
  """
  y1: real, [batch, time, feature_dim]
  y2: real, [batch, time, feature_dim]
  """
  loss = tf.square(y1-y2)
  loss = tf.reduce_mean(tf.reduce_sum(loss, 0))
  return loss

def batch_time_fea_complex_mse(y1, y2):
  """
  y1: complex, [batch, time, feature_dim]
  y2: conplex, [batch, time, feature_dim]
  """
  y1_real = tf.math.real(y1)
  y1_imag = tf.math.imag(y1)
  y2_real = tf.math.real(y2)
  y2_imag = tf.math.imag(y2)
  loss_real = batch_time_fea_real_mse(y1_real, y2_real)
  loss_imag = batch_time_fea_real_mse(y1_imag, y2_imag)
  loss = loss_real + loss_imag
  return loss

def batch_real_relativeMSE(y1, y2, RL_epsilon, index_=2.0):
  # y1, y2 : [batch, time, feature]
  # refer_sum = tf.maximum(tf.abs(y1)+tf.abs(y2),1e-12)
  # small_val_debuff = tf.pow(refer_sum*RL_epsilon*1.0,-1.0)+1.0-tf.pow(RL_epsilon*1.0,-1.0)
  # relative_loss = tf.abs(y1-y2)/refer_sum/small_val_debuff
  relative_loss = tf.abs(y1-y2)/(tf.abs(y1)+tf.abs(y2)+RL_epsilon)
  cost = tf.reduce_mean(tf.reduce_sum(tf.pow(relative_loss, index_), 0))
  return cost

def batch_complex_relativeMSE(y1, y2, RL_epsilon, index_=2.0):
  """
  y1: complex, [batch, time, feature_dim]
  y2: conplex, [batch, time, feature_dim]
  """
  y1_real = tf.math.real(y1)
  y1_imag = tf.math.imag(y1)
  y2_real = tf.math.real(y2)
  y2_imag = tf.math.imag(y2)
  loss_real = batch_real_relativeMSE(y1_real, y2_real, RL_epsilon)
  loss_imag = batch_real_relativeMSE(y1_imag, y2_imag, RL_epsilon)
  loss = 0.5*loss_real+0.5*loss_imag
  return loss

def batch_wav_L1_loss(y1, y2):
  loss = tf.reduce_mean(tf.reduce_sum(tf.abs(y1-y2), 0))
  return loss

def batch_wav_L2_loss(y1, y2):
  loss = tf.reduce_mean(tf.reduce_sum(tf.square(y1-y2), 0))
  return loss

def batch_wav_relativeMSE(y1, y2, RL_epsilon, index_=2.0):
  loss = batch_real_relativeMSE(y1, y2, RL_epsilon, index_=index_)
  return loss

def batch_CosSim_loss(est, ref): # -cos
  '''
  est, ref: [batch, ..., n_sample]
  '''
  cos_sim = - tf.divide(vec_dot_mul(est, ref), # [batch, ...]
                        tf.multiply(vec_normal(est), vec_normal(ref)))
  loss = tf.reduce_sum(cos_sim, 0) # [...,]
  return loss

def batch_SquareCosSim_loss(est, ref): # -cos^2
  loss_s1 = - tf.divide(tf.square(vec_dot_mul(est, ref)),  # [batch, ...]
                        tf.multiply(vec_dot_mul(est, est),
                                    vec_dot_mul(ref, ref)))
  loss = tf.reduce_sum(loss_s1, 0)
  return loss

def batch_short_time_CosSim_loss(est, ref, st_frame_length, st_frame_step): # -cos
  st_est = tf.signal.frame(est, frame_length=st_frame_length, # [batch, frame, st_wav]
                           frame_step=st_frame_step, pad_end=True)
  st_ref = tf.signal.frame(ref, frame_length=st_frame_length,
                           frame_step=st_frame_step, pad_end=True)
  loss_s1 = batch_CosSim_loss(st_est, st_ref)  # [frame]
  loss = tf.reduce_mean(loss_s1, 0)
  return loss

def batch_short_time_SquareCosSim_loss(est, ref, st_frame_length, st_frame_step): # -cos^2
  st_est = tf.signal.frame(est, frame_length=st_frame_length, # [batch, frame, st_wav]
                           frame_step=st_frame_step, pad_end=True)
  st_ref = tf.signal.frame(ref, frame_length=st_frame_length,
                           frame_step=st_frame_step, pad_end=True)
  loss_s1 = batch_SquareCosSim_loss(st_est, st_ref)  # [frame]
  loss = tf.reduce_mean(loss_s1, 0)
  return loss
