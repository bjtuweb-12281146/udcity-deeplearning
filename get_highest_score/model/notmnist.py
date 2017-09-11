# tensorflow importings
import tensorflow as tf
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util


def variable_on_cpu(name,initializer):
    with tf.device("/cpu:0"):
        return tf.Variable(name=name, initial_value=initializer)


def variable_on_cpu_wd(name,initializer,wd=None):
    var = variable_on_cpu(name,initializer)
    if wd is not None:
        l2_loss = tf.multiply(tf.nn.l2_loss(var), wd)
        tf.add_to_collection("losses", l2_loss)
    return var

def variable_summaries(var,name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    def __init__(self,
                 batch_size,
                 every_n_steps=10,
                 every_n_seconds=None):
        # pdb.set_trace()
        if (every_n_seconds is None) == (every_n_steps is None):
            raise  ValueError("exactly one of every_n_seteps and "
                              " every_n_seconds should be provided")
        self._timer = tf.train.SecondOrStepTimer(
            every_steps=every_n_steps,every_secs=every_n_seconds
        )
        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use StepCounterHook.')

    def before_run(self,run_context):
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context
        global_step = run_values.results
        # logging.error(global_step)
        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                global_step)
            # pdb.set_trace()
            # logging.error("total steps: {}".format(self._total_steps))
            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = self._batch_size * (
                    self._total_steps / self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                # Average examples/sec followed by current examples/sec
                tf.logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                             average_examples_per_sec, current_examples_per_sec,
                             self._total_steps)

