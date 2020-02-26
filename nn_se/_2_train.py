import tensorflow as tf
import os
import numpy as np
import time
import collections
from pathlib import Path
import sys


from .models import model_builder
from .models import modules
from .dataloader import dataloader
from .utils import misc_utils
from .FLAGS import PARAM


def __relative_impr(prev_, new_, declining=False):
  if declining:
    return (prev_-new_)/(abs(prev_)+abs(new_)+1e-8)
  return (new_-prev_)/(abs(prev_)+abs(new_)+1e-8)


class TrainOutputs(
    collections.namedtuple("TrainOutputs",
                           ("sum_loss", "stop_c_loss", "show_losses", "cost_time", "lr"))):
  pass


def train_one_epoch(sess, train_model, train_log_file):
  i, lr = 0, -1
  s_time = time.time()
  minbatch_time = time.time()
  one_batch_time = time.time()


  avg_sum_loss = None
  avg_show_losses = None
  avg_stop_c_loss = None
  i = 0
  total_i = PARAM.n_train_set_records//PARAM.batch_size
  while True:
    try:
      _, lr, sum_loss, show_losses, stop_c_loss = sess.run(
          [train_model.train_op,
           train_model.lr,
           train_model.losses.sum_loss,
           train_model.losses.show_losses,
           train_model.losses.stop_criterion_loss])

      if avg_sum_loss is None:
        avg_sum_loss = sum_loss
        avg_show_losses = show_losses
        avg_stop_c_loss = stop_c_loss
      else:
        avg_sum_loss += sum_loss
        avg_show_losses += show_losses
        avg_stop_c_loss += stop_c_loss
      i += 1
      print("\r", end="")
      print("train: %d/%d, cost %.2fs, sum_loss %.4f, stop_loss %.4f, show_losses %s  " % (
            i, total_i, time.time()-one_batch_time, sum_loss, stop_c_loss,
            str(np.round(show_losses, 4))),
            flush=True, end="")
      one_batch_time = time.time()
      if i % PARAM.batches_to_logging == 0:
        print("\r", end="")
        msg = "     Minbatch %04d: sum_loss:%.4f, stop_loss:%.4f, show_losses:%s, lr:%.2e, time:%ds. \n" % (
                i, avg_sum_loss/i, stop_c_loss/i, np.round(show_losses/i, 4), lr, time.time()-minbatch_time,
              )
        minbatch_time = time.time()
        misc_utils.print_log(msg, train_log_file)
    except tf.errors.OutOfRangeError:
      break
  print("\r", end="")
  e_time = time.time()
  avg_sum_loss = avg_sum_loss / total_i
  avg_show_losses = avg_show_losses / total_i
  avg_stop_c_loss = avg_stop_c_loss / total_i
  return TrainOutputs(sum_loss=avg_sum_loss,
                      stop_c_loss=avg_stop_c_loss,
                      show_losses=np.round(avg_show_losses, 4),
                      cost_time=e_time-s_time,
                      lr=lr)


class EvalOutputs(
    collections.namedtuple("EvalOutputs",
                           ("sum_loss", "show_losses",
                            "stop_criterion_loss", "cost_time"))):
  pass

def round_lists(lst, rd):
  return [round(n,rd) if type(n) is not list else round_lists(n,rd) for n in lst]

def unfold_list(lst):
  ans_lst = []
  [ans_lst.append(n) if type(n) is not list else ans_lst.extend(unfold_list(n)) for n in lst]
  return ans_lst

def eval_one_epoch(sess, val_model):
  val_s_time = time.time()
  ont_batch_time = time.time()

  avg_sum_loss = None
  avg_show_losses = None
  avg_stop_c_loss = None
  i = 0
  total_i = PARAM.n_val_set_records//PARAM.batch_size
  while True:
    try:
      sum_loss, show_losses, stop_c_loss = sess.run(
          [val_model.losses.sum_loss,
           val_model.losses.show_losses,
           val_model.losses.stop_criterion_loss])

      if avg_sum_loss is None:
        avg_sum_loss = sum_loss
        avg_show_losses = show_losses
        avg_stop_c_loss = stop_c_loss
      else:
        avg_sum_loss += sum_loss
        avg_show_losses += show_losses
        avg_stop_c_loss += stop_c_loss
      i += 1
      print("\r", end="")
      print("validate: %d/%d, cost %.2fs, sum_loss %.4f, stop_loss %.4f, show_losses %s"
            "          " % (
                i, total_i, time.time()-ont_batch_time, sum_loss, stop_c_loss,
                str(np.round(show_losses, 4))
            ),
            flush=True, end="")
      ont_batch_time = time.time()
    except tf.errors.OutOfRangeError:
      break

  print("\r", end="")
  avg_sum_loss = avg_sum_loss / total_i
  avg_show_losses = avg_show_losses / total_i
  avg_stop_c_loss = avg_stop_c_loss / total_i
  val_e_time = time.time()
  return EvalOutputs(sum_loss=avg_sum_loss,
                     show_losses=np.round(avg_show_losses, 4),
                     stop_criterion_loss=avg_stop_c_loss,
                     cost_time=val_e_time-val_s_time)


def main():
  train_log_file = misc_utils.train_log_file_dir()
  ckpt_dir = misc_utils.ckpt_dir()
  hparam_file = misc_utils.hparams_file_dir()
  if not train_log_file.parent.exists():
    os.makedirs(str(train_log_file.parent))
  if not ckpt_dir.exists():
    os.mkdir(str(ckpt_dir))

  misc_utils.save_hparams(str(hparam_file))

  g = tf.Graph()
  with g.as_default():
    with tf.name_scope("inputs"):
      noisy_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.train_noisy_set)
      clean_trainset_wav = misc_utils.datasets_dir().joinpath(PARAM.train_clean_set)
      noisy_valset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_noisy_set)
      clean_valset_wav = misc_utils.datasets_dir().joinpath(PARAM.validation_clean_set)
      train_inputs = dataloader.get_batch_inputs_from_nosiyCleanDataset(noisy_trainset_wav,
                                                                        clean_trainset_wav)
      val_inputs = dataloader.get_batch_inputs_from_nosiyCleanDataset(noisy_valset_wav,
                                                                      clean_valset_wav)

    ModelC = model_builder.get_model_class_and_var()

    train_model = ModelC(PARAM.MODEL_TRAIN_KEY, train_inputs.mixed, train_inputs.clean)
    # tf.compat.v1.get_variable_scope().reuse_variables()
    val_model = ModelC(PARAM.MODEL_VALIDATE_KEY, val_inputs.mixed,val_inputs.clean)
    init = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
    # misc_utils.show_variables(train_model.save_variables)
    # misc_utils.show_variables(val_model.save_variables)
  g.finalize()

  config = tf.compat.v1.ConfigProto()
  # config.gpu_options.allow_growth = PARAM.GPU_RAM_ALLOW_GROWTH
  config.gpu_options.per_process_gpu_memory_fraction = PARAM.GPU_PARTION
  config.allow_soft_placement = False
  sess = tf.compat.v1.Session(config=config, graph=g)
  sess.run(init)

  # region validation before training
  sess.run(val_inputs.initializer)
  misc_utils.print_log("stop criterion losses: "+str(PARAM.stop_criterion_losses)+"\n", train_log_file)
  misc_utils.print_log("show losses: "+str(PARAM.show_losses)+"\n", train_log_file)
  evalOutputs_prev = eval_one_epoch(sess, val_model)
  misc_utils.print_log("                                            "
                       "                                            "
                       "                                         \n\n",
                       train_log_file, no_time=True)
  val_msg = "PRERUN.val> sum_loss:%.4F, stop_loss:%.4F, show_losses:%s, Cost itme:%.2Fs.\n" % (
      evalOutputs_prev.sum_loss,
      evalOutputs_prev.stop_criterion_loss,
      evalOutputs_prev.show_losses,
      evalOutputs_prev.cost_time)
  misc_utils.print_log(val_msg, train_log_file)

  assert PARAM.s_epoch > 0, 'start epoch > 0 is required.'
  model_abandon_time = 0

  for epoch in range(PARAM.s_epoch, PARAM.max_epoch+1):
    misc_utils.print_log("\n\n", train_log_file, no_time=True)
    misc_utils.print_log("  Epoch %03d:\n" % epoch, train_log_file)
    misc_utils.print_log("   sum_losses: "+str(PARAM.sum_losses)+"\n", train_log_file)
    misc_utils.print_log("   stop_criterion_losses: "+str(PARAM.stop_criterion_losses)+"\n", train_log_file)
    misc_utils.print_log("   show_losses: "+str(PARAM.show_losses)+"\n", train_log_file)

    # train
    sess.run(train_inputs.initializer)
    trainOutputs = train_one_epoch(sess, train_model, train_log_file)
    misc_utils.print_log("     Train     > sum_loss:%.4f, stop_loss:%.4f, show_losses:%s, lr:%.2e Time:%ds.   \n" % (
        trainOutputs.sum_loss,
        trainOutputs.stop_c_loss,
        trainOutputs.show_losses,
        trainOutputs.lr,
        trainOutputs.cost_time),
        train_log_file)

    # validation
    sess.run(val_inputs.initializer)
    evalOutputs = eval_one_epoch(sess, val_model)
    val_loss_rel_impr = __relative_impr(evalOutputs_prev.avg_loss, evalOutputs.avg_loss, True)
    misc_utils.print_log("     Validation> sum_loss%.4f, stop_loss:%.4f, show_losses:%s, Time:%ds.           \n" % (
        evalOutputs.sum_loss,
        evalOutputs.stop_criterion_loss,
        evalOutputs.show_losses,
        evalOutputs.cost_time),
        train_log_file)

    # save or abandon ckpt
    ckpt_name = PARAM().config_name()+('_iter%04d_trloss%.4f_valloss%.4f_lr%.2e_duration%ds' % (
        epoch, trainOutputs.sum_loss, evalOutputs.sum_loss, trainOutputs.lr,
        trainOutputs.cost_time+evalOutputs.cost_time))
    if val_loss_rel_impr > 0 or PARAM.no_abandon:
      train_model.saver.save(sess, str(ckpt_dir.joinpath(ckpt_name)))
      evalOutputs_prev = evalOutputs
      best_ckpt_name = ckpt_name
      msg = "     ckpt(%s) saved.\n" % ckpt_name
    else:
      model_abandon_time += 1
      # tf.compat.v1.logging.set_verbosity(tf.logging.WARN)
      train_model.saver.restore(sess,
                                str(ckpt_dir.joinpath(best_ckpt_name)))
      # tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
      msg = "     ckpt(%s) abandoned.\n" % ckpt_name
    misc_utils.print_log(msg, train_log_file)

    # start lr halving
    if val_loss_rel_impr < PARAM.start_halving_impr and (not PARAM.use_lr_warmup):
      new_lr = trainOutputs.lr * PARAM.lr_halving_rate
      train_model.change_lr(sess, new_lr)

    # stop criterion
    if (epoch >= PARAM.max_epoch or
            model_abandon_time >= PARAM.max_model_abandon_time):
      misc_utils.print_log("\n\n", train_log_file, no_time=True)
      msg = "Training finished, final learning rate %e.\n" % trainOutputs.lr
      tf.logging.info(msg)
      misc_utils.print_log(msg, train_log_file)
      break

  sess.close()
  misc_utils.print_log("\n", train_log_file, no_time=True)
  msg = '################### Training Done. ###################\n'
  misc_utils.print_log(msg, train_log_file)


if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])
  main()
  """
  run cmd:
  `CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m xx._2_train`
  """
