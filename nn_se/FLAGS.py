class StaticKey(object):
  MODEL_TRAIN_KEY = 'train'
  MODEL_VALIDATE_KEY = 'val'
  MODEL_INFER_KEY = 'infer'

  # dataset name
  train_name="train"
  validation_name="validation"
  test_name="test"

  def config_name(self): # config_name
    return self.__class__.__name__

class BaseConfig(StaticKey):
  VISIBLE_GPU = "0"
  root_dir = '/home/lhf/worklhf/PHASEN/'
  # datasets_name = 'vctk_musan_datasets'
  datasets_name = 'noisy_datasets_16k'
  '''
  # dir to store log, model and results files:
  $root_dir/$datasets_name: datasets dir
  $root_dir/exp/$config_name/log: logs(include tensorboard log)
  $root_dir/exp/$config_name/ckpt: ckpt
  $root_dir/exp/$config_name/enhanced_testsets: enhanced results
  $root_dir/exp/$config_name/hparams
  '''

  min_TF_version = "1.14.0"


  train_noisy_set = 'noisy_trainset_wav'
  train_clean_set = 'clean_trainset_wav'
  validation_noisy_set = 'noisy_testset_wav'
  validation_clean_set = 'clean_testset_wav'
  test_noisy_sets = ['noisy_testset_wav']
  test_clean_sets = ['clean_testset_wav']

  n_train_set_records = 11572
  n_val_set_records = 824
  n_test_set_records = 824

  train_val_wav_seconds = 3.0

  batch_size = 12
  n_processor_tfdata = 4

  model_name = "PHASEN"

  relative_loss_epsilon = 0.1
  RL_idx = 2.0
  st_frame_length_for_loss = 512
  st_frame_step_for_loss = 128
  sampling_rate = 16000
  frame_length = 400
  frame_step = 160
  fft_length = 512
  max_keep_ckpt = 30
  optimizer = "Adam" # "Adam" | "RMSProp"
  learning_rate = 0.0005
  max_gradient_norm = 5.0

  GPU_RAM_ALLOW_GROWTH = True
  GPU_PARTION = 0.45

  s_epoch = 1
  max_epoch = 40
  batches_to_logging = 300

  max_model_abandon_time = 3
  no_abandon = True
  use_lr_warmup = True # true: lr warmup; false: lr halving
  warmup_steps = 6000. # for (use_lr_warmup == true)
  start_halving_impr = 0.01 # no use for (use_lr_warmup == true)
  lr_halving_rate = 0.7 # no use for (use_lr_warmup == true)

  # melMat: tf.contrib.signal.linear_to_mel_weight_matrix(129,129,8000,125,3900)
  # plt.pcolormesh
  # import matplotlib.pyplot as plt

  """
  @param not_transformed_losses/transformed_losses[add FT before loss_name]:
  loss_mag_mse, loss_spec_mse, loss_wav_L1, loss_wav_L2,
  loss_mag_reMse, loss_reSpecMse, loss_reWavL2,
  loss_sdrV1, loss_sdrV2, loss_stSDRV3, loss_cosSimV1, loss_cosSimV2,
  """
  sum_losses = ["loss_compressedMag_mse", "loss_complexPhase_mse"]
  sum_losses_w = []
  show_losses = ["loss_compressedMag_mse", "loss_complexPhase_mse"]
  show_losses_w = []
  stop_criterion_losses = ["loss_compressedMag_mse", "loss_complexPhase_mse"]
  stop_criterion_losses_w = []

  # just for "DISCRIMINATOR_AD_MODEL"

  channel_A = 32
  channel_P = 16
  prenet_A_kernels = [[1,7], [7, 1]]
  prenet_P_kernels = [[5,3], [25,1]]
  n_TSB = 3
  frequency_dim = 257
  loss_compressedMag_idx = 0.3



class p40(BaseConfig):
  n_processor_gen_tfrecords = 56
  n_processor_tfdata = 8
  GPU_PARTION = 0.225
  root_dir = '/home/zhangwenbo5/lihongfeng/PHASEN'


class se_magMSE(p40): # done p40
  '''
  mag mse
  '''
  GPU_PARTION = 0.23
  losses_position = ['not_transformed_losses']
  not_transformed_losses = ['loss_mag_mse']
  # relative_loss_epsilon = 0.1
  blstm_layers = 3
  # transformed_losses = ['FTloss_mag_mse']
  # FT_type = ["LogValueT"]
  # add_FeatureTrans_in_SE_inputs = False

  stop_criterion_losses = ['loss_mag_mse']
  show_losses = ['loss_mag_mse']

PARAM = se_magMSE

# CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 python -m xxx._2_train
