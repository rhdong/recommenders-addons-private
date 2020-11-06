# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""unit tests of dynamic embedding ops
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import math
import numpy as np
import os
import six
import tempfile

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras import initializers as keras_init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import device_setter
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat


# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
def _type_converter(tf_type):
  mapper = {
    dtypes.int32: np.int32,
    dtypes.int64: np.int64,
    dtypes.float32: np.float,
    dtypes.float64: np.float64
  }
  return mapper[tf_type]


def _get_devices():
  return ['/gpu:0' if test_util.is_gpu_available() else '/cpu:0']


def _check_device(op, expexted_device='gpu'):
  return expexted_device.upper() in op.device


def embedding_result(params, id_vals, weight_vals=None):
  if weight_vals is None:
    weight_vals = np.copy(id_vals)
    weight_vals.fill(1)
  values = []
  weights = []
  weights_squared = []
  for pms, ids, wts in zip(params, id_vals, weight_vals):
    value_aggregation = None
    weight_aggregation = None
    squared_weight_aggregation = None
    if isinstance(ids, compat.integral_types):
      pms = [pms]
      ids = [ids]
      wts = [wts]
    for val, i, weight_value in zip(pms, ids, wts):
      if value_aggregation is None:
        assert weight_aggregation is None
        assert squared_weight_aggregation is None
        value_aggregation = val * weight_value
        weight_aggregation = weight_value
        squared_weight_aggregation = weight_value * weight_value
      else:
        assert weight_aggregation is not None
        assert squared_weight_aggregation is not None
        value_aggregation += val * weight_value
        weight_aggregation += weight_value
        squared_weight_aggregation += weight_value * weight_value
    values.append(value_aggregation)
    weights.append(weight_aggregation)
    weights_squared.append(squared_weight_aggregation)
  values = np.array(values).astype(np.float32)
  weights = np.array(weights).astype(np.float32)
  weights_squared = np.array(weights_squared).astype(np.float32)
  return values, weights, weights_squared


def ids_and_weights_2d(embed_dim=4):
  # Each row demonstrates a test case:
  #   Row 0: multiple valid ids, 1 invalid id, weighted mean
  #   Row 1: all ids are invalid (leaving no valid ids after pruning)
  #   Row 2: no ids to begin with
  #   Row 3: single id
  #   Row 4: all ids have <=0 weight
  indices = [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [4, 0], [4, 1]]
  ids = [0, 1, -1, -1, 2, 0, 1]
  weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
  shape = [5, embed_dim]

  sparse_ids = sparse_tensor.SparseTensor(
    constant_op.constant(indices, dtypes.int64),
    constant_op.constant(ids, dtypes.int64),
    constant_op.constant(shape, dtypes.int64))

  sparse_weights = sparse_tensor.SparseTensor(
    constant_op.constant(indices, dtypes.int64),
    constant_op.constant(weights, dtypes.float32),
    constant_op.constant(shape, dtypes.int64))

  return sparse_ids, sparse_weights


def ids_and_weights_3d(embed_dim=4):
  # Each (2-D) index demonstrates a test case:
  #   Index 0, 0: multiple valid ids, 1 invalid id, weighted mean
  #   Index 0, 1: all ids are invalid (leaving no valid ids after pruning)
  #   Index 0, 2: no ids to begin with
  #   Index 1, 0: single id
  #   Index 1, 1: all ids have <=0 weight
  #   Index 1, 2: no ids to begin with
  indices = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [1, 0, 0], [1, 1, 0],
             [1, 1, 1]]
  ids = [0, 1, -1, -1, 2, 0, 1]
  weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
  shape = [2, 3, embed_dim]

  sparse_ids = sparse_tensor.SparseTensor(
    constant_op.constant(indices, dtypes.int64),
    constant_op.constant(ids, dtypes.int64),
    constant_op.constant(shape, dtypes.int64))

  sparse_weights = sparse_tensor.SparseTensor(
    constant_op.constant(indices, dtypes.int64),
    constant_op.constant(weights, dtypes.float32),
    constant_op.constant(shape, dtypes.int64))

  return sparse_ids, sparse_weights


def _random_weights(key_dtype=dtypes.int64,
                    value_dtype=dtypes.float32,
                    vocab_size=4,
                    embed_dim=4,
                    num_shards=1):
  assert vocab_size > 0
  assert embed_dim > 0
  assert num_shards > 0
  assert num_shards <= vocab_size

  initializer = init_ops.truncated_normal_initializer(mean=0.0,
                                                      stddev=1.0 /
                                                             math.sqrt(
                                                               vocab_size),
                                                      dtype=dtypes.float32)
  embedding_weights = de.get_variable(key_dtype=key_dtype,
                                      value_dtype=value_dtype,
                                      devices=_get_devices() * num_shards,
                                      name="embedding_weights",
                                      initializer=initializer,
                                      dim=embed_dim)
  return embedding_weights


def _test_dir(temp_dir, test_name):
  """Create an empty dir to use for tests.

  Args:
    temp_dir: Tmp directory path.
    test_name: Name of the test.

  Returns:
    Absolute path to the test directory.
  """
  test_dir = os.path.join(temp_dir, test_name)
  if os.path.isdir(test_dir):
    for f in glob.glob('%s/*' % test_dir):
      os.remove(f)
  else:
    os.makedirs(test_dir)
  return test_dir


def _create_dynamic_shape_tensor(max_len=100,
                                 min_len=2,
                                 min_val=0x0000f00000000001,
                                 max_val=0x0000f00000000020,
                                 dtype=np.int64):
  def _func():
    length = np.random.randint(min_len, max_len)
    tensor = np.random.randint(min_val, max_val, max_len, dtype=dtype)
    tensor = np.array(tensor[0:length], dtype=dtype)
    return tensor

  return _func


default_config = config_pb2.ConfigProto(
  allow_soft_placement=False,
  gpu_options=config_pb2.GPUOptions(allow_growth=True))


@test_util.run_all_in_graph_and_eager_modes
class DynamicEmbeddingOpTest(test.TestCase):

  def test_variable(self):
    id = 0
    if test_util.is_gpu_available():
      dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 256, 500]
    else:
      dim_list = [1, 10]
    for key_dtype, value_dtype, dim in itertools.product(
        [dtypes.int64], [dtypes.float32], dim_list):
      id += 1
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        keys = constant_op.constant([0, 1, 2, 3], key_dtype)
        values = constant_op.constant(
          [[0] * dim, [1] * dim, [2] * dim, [3] * dim], value_dtype)
        table = de.get_variable('t1-' + str(id),
                                key_dtype=key_dtype,
                                value_dtype=value_dtype,
                                initializer=-1,
                                dim=dim)
        self.assertAllEqual(0, self.evaluate(table.size()))

        self.evaluate(table.upsert(keys, values))
        self.assertAllEqual(4, self.evaluate(table.size()))

        remove_keys = constant_op.constant([1, 5], key_dtype)
        self.evaluate(table.remove(remove_keys))
        self.assertAllEqual(3, self.evaluate(table.size()))

        remove_keys = constant_op.constant([0, 1, 5], key_dtype)
        output = table.lookup(remove_keys)
        self.assertAllEqual([3, dim], output.get_shape())

        result = self.evaluate(output)
        self.assertAllEqual([[0] * dim, [-1] * dim, [-1] * dim], result)

        exported_keys, exported_values = table.export()

        # exported data is in the order of the internal map, i.e. undefined
        sorted_keys = np.sort(self.evaluate(exported_keys))
        sorted_values = np.sort(self.evaluate(exported_values), axis=0)
        self.assertAllEqual([0, 2, 3], sorted_keys)
        self.assertAllEqual([[0] * dim, [2] * dim, [3] * dim], sorted_values)

        del table

  def test_variable_initializer(self):
    id = 0
    for initializer, target_mean, target_stddev in [
      (-1.0, -1.0, 0.0),
      (init_ops.random_normal_initializer(0.0, 0.01), 0.0, 0.01),
    ]:
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()):
        id += 1
        keys = constant_op.constant(list(range(2 ** 17)), dtypes.int64)
        table = de.get_variable('t1' + str(id),
                                key_dtype=dtypes.int64,
                                value_dtype=dtypes.float32,
                                initializer=initializer,
                                dim=10)
        vals_op = table.lookup(keys)
        mean = self.evaluate(math_ops.reduce_mean(vals_op))
        stddev = self.evaluate(math_ops.reduce_std(vals_op))
        rtol = 2e-5
        atol = rtol
        self.assertAllClose(target_mean, mean, rtol, atol)
        self.assertAllClose(target_stddev, stddev, rtol, atol)

  def test_save_restore(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0.], [1.], [2.]], dtypes.float32)
      table = de.Variable(key_dtype=dtypes.int64,
                          value_dtype=dtypes.float32,
                          initializer=-1.,
                          name='t1',
                          dim=1)

      save = saver.Saver(var_list=[v0, v1, table])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)

      del table

    with self.session(config=default_config, graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = -1
      table = de.Variable(name="t1",
                          key_dtype=dtypes.int64,
                          value_dtype=dtypes.float32,
                          initializer=-1.,
                          dim=1,
                          checkpoint=True)
      self.evaluate(
        table.upsert(constant_op.constant([0, 1], dtypes.int64),
                     constant_op.constant([[12.], [24.]], dtypes.float32)))
      size_op = table.size()
      self.assertAllEqual(2, self.evaluate(size_op))

      save = saver.Saver(var_list=[v0, v1, table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual([10.0], self.evaluate(v0))
      self.assertEqual([20.0], self.evaluate(v1))

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([5, 0, 1, 2, 6], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[-1.], [0.], [1.], [2.], [-1.]],
                          self.evaluate(output))

      del table

  def test_save_restore_only_table(self):
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(config=default_config,
                      graph=ops.Graph(),
                      use_gpu=test_util.is_gpu_available()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.Variable(dtypes.int64,
                          dtypes.int32,
                          name='t1',
                          initializer=default_val,
                          checkpoint=True)

      save = saver.Saver([table])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, six.string_types)
      self.assertEqual(save_path, val)
      del table

    with self.session(config=default_config,
                      graph=ops.Graph(),
                      use_gpu=test_util.is_gpu_available()) as sess:
      default_val = -1
      table = de.Variable(dtypes.int64,
                          dtypes.int32,
                          name='t1',
                          initializer=default_val,
                          checkpoint=True)
      self.evaluate(
        table.upsert(constant_op.constant([0, 2], dtypes.int64),
                     constant_op.constant([[12], [24]], dtypes.int32)))
      self.assertAllEqual(2, self.evaluate(table.size()))

      save = saver.Saver([table._tables[0]])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.

      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([[0], [1], [2], [-1], [-1]], self.evaluate(output))
      del table

  def test_traing_save_restore(self):
    opt = de.DynamicEmbeddingOptimizer(adam.AdamOptimizer(0.3))
    id = 0
    if test_util.is_gpu_available():
      dim_list = [1, 2, 4, 8, 10, 16, 32, 64, 100, 256, 500]
    else:
      dim_list = [10]
    for key_dtype, value_dtype, dim, step in itertools.product(
        [dtypes.int64],
        [dtypes.float32],
        dim_list,
        [10],
    ):
      id += 1
      save_dir = os.path.join(self.get_temp_dir(), "save_restore")
      save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

      ids = script_ops.py_func(_create_dynamic_shape_tensor(),
                               inp=[],
                               Tout=key_dtype,
                               stateful=True)

      params = de.get_variable(name="params-test-0915-" + str(id),
                               key_dtype=key_dtype,
                               value_dtype=value_dtype,
                               initializer=init_ops.random_normal_initializer(
                                 0.0, 0.01),
                               dim=dim)
      _, var0 = de.embedding_lookup(params, ids, return_trainable=True)
      loss = lambda: var0 * var0

      params_keys, params_vals = params.export()
      mini = opt.minimize(loss, var_list=[var0])
      opt_slots = [opt.get_slot(var0, _s) for _s in opt.get_slot_names()]
      _saver = saver.Saver([params] + [_s.params for _s in opt_slots])

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        self.evaluate(variables.global_variables_initializer())
        for _i in range(step):
          self.evaluate([mini])
        size_before_saved = self.evaluate(params.size())
        np_params_keys_before_saved = self.evaluate(params_keys)
        np_params_vals_before_saved = self.evaluate(params_vals)
        opt_slots_kv_pairs = [_s.params.export() for _s in opt_slots]
        np_slots_kv_pairs_before_saved = [
          self.evaluate(_kv) for _kv in opt_slots_kv_pairs
        ]
        _saver.save(sess, save_path)

      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()) as sess:
        self.evaluate(variables.global_variables_initializer())
        self.assertAllEqual(0, self.evaluate(params.size()))

        _saver.restore(sess, save_path)
        params_keys_restored, params_vals_restored = params.export()
        size_after_restored = self.evaluate(params.size())
        np_params_keys_after_restored = self.evaluate(params_keys_restored)
        np_params_vals_after_restored = self.evaluate(params_vals_restored)

        opt_slots_kv_pairs_restored = [_s.params.export() for _s in opt_slots]
        np_slots_kv_pairs_after_restored = [
          self.evaluate(_kv) for _kv in opt_slots_kv_pairs_restored
        ]
        self.assertAllEqual(size_before_saved, size_after_restored)
        self.assertAllEqual(np.sort(np_params_keys_before_saved),
                            np.sort(np_params_keys_after_restored))
        self.assertAllEqual(np.sort(np_params_vals_before_saved, axis=0),
                            np.sort(np_params_vals_after_restored, axis=0))
        for pairs_before, pairs_after in zip(np_slots_kv_pairs_before_saved,
                                             np_slots_kv_pairs_after_restored):
          self.assertAllEqual(np.sort(pairs_before[0], axis=0),
                              np.sort(pairs_after[0], axis=0))
          self.assertAllEqual(np.sort(pairs_before[1], axis=0),
                              np.sort(pairs_after[1], axis=0))
        if test_util.is_gpu_available():
          self.assertTrue('GPU' in params.tables[0].resource_handle.device)

  def test_get_variable(self):
    with self.session(config=default_config,
                      graph=ops.Graph(),
                      use_gpu=test_util.is_gpu_available()) as sess:
      default_val = -1
      with variable_scope.variable_scope("embedding", reuse=True):
        table1 = de.get_variable("t1",
                                 dtypes.int64,
                                 dtypes.int32,
                                 initializer=default_val,
                                 dim=2)
        table2 = de.get_variable("t1",
                                 dtypes.int64,
                                 dtypes.int32,
                                 initializer=default_val,
                                 dim=2)
        table3 = de.get_variable("t2",
                                 dtypes.int64,
                                 dtypes.int32,
                                 initializer=default_val,
                                 dim=2)

      self.assertAllEqual(table1, table2)
      self.assertNotEqual(table1, table3)

  def test_get_variable_reuse_error(self):
    ops.disable_eager_execution()
    with self.session(config=default_config,
                      graph=ops.Graph(),
                      use_gpu=test_util.is_gpu_available()) as sess:
      with variable_scope.variable_scope("embedding", reuse=False):
        _ = de.get_variable("t900", initializer=-1, dim=2)
        with self.assertRaisesRegexp(ValueError,
                                     'Variable embedding/t900 already exists'):
          _ = de.get_variable("t900", initializer=-1, dim=2)

  @test_util.run_v1_only("Multiple sessions")
  def test_sharing_between_multi_sessions(self):
    ops.disable_eager_execution()
    # Start a server to store the table state
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target, config=default_config)
    session2 = session.Session(server.target, config=default_config)

    table = de.get_variable("tx100",
                            dtypes.int64,
                            dtypes.int32,
                            initializer=0,
                            dim=1)

    # Populate the table in the first session
    with session1:
      with ops.device(_get_devices()[0]):
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(variables.local_variables_initializer())
        self.assertAllEqual(0, table.size().eval())

        keys = constant_op.constant([11, 12], dtypes.int64)
        values = constant_op.constant([[11], [12]], dtypes.int32)
        table.upsert(keys, values).run()
        self.assertAllEqual(2, table.size().eval())

        output = table.lookup(constant_op.constant([11, 12, 13], dtypes.int64))
        self.assertAllEqual([[11], [12], [0]], output.eval())

    # Verify that we can access the shared data from the second session
    with session2:
      with ops.device(_get_devices()[0]):
        self.assertAllEqual(2, table.size().eval())

        output = table.lookup(constant_op.constant([10, 11, 12], dtypes.int64))
        self.assertAllEqual([[0], [11], [12]], output.eval())

  def test_dynamic_embedding_variable(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [4, 5], [6, 7]],
                                    dtypes.int32)
      table = de.get_variable('t10',
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val,
                              dim=2)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([3, 4], dtypes.int64)
      self.evaluate(table.remove(remove_keys))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 4], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([3, 2], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([[0, 1], [2, 3], [-1, -1]], result)

      exported_keys, exported_values = table.export()
      # exported data is in the order of the internal map, i.e. undefined
      sorted_keys = np.sort(self.evaluate(exported_keys))
      sorted_values = np.sort(self.evaluate(exported_values), axis=0)
      self.assertAllEqual([0, 1, 2], sorted_keys)
      sorted_expected_values = np.sort([[4, 5], [2, 3], [0, 1]], axis=0)
      self.assertAllEqual(sorted_expected_values, sorted_values)

      del table

  def test_dynamic_embedding_variable_export_insert(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int32)
      table1 = de.get_variable("t101",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val,
                               dim=2)
      self.assertAllEqual(0, self.evaluate(table1.size()))
      self.evaluate(table1.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table1.size()))

      input_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      expected_output = [[0, 1], [2, 3], [-1, -1]]
      output1 = table1.lookup(input_keys)
      self.assertAllEqual(expected_output, self.evaluate(output1))

      exported_keys, exported_values = table1.export()
      self.assertAllEqual(3, self.evaluate(exported_keys).size)
      self.assertAllEqual(6, self.evaluate(exported_values).size)

      # Populate a second table from the exported data
      table2 = de.get_variable("t102",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val,
                               dim=2)
      self.assertAllEqual(0, self.evaluate(table2.size()))
      self.evaluate(table2.upsert(exported_keys, exported_values))
      self.assertAllEqual(3, self.evaluate(table2.size()))

      # Verify lookup result is still the same
      output2 = table2.lookup(input_keys)
      self.assertAllEqual(expected_output, self.evaluate(output2))

  def test_dynamic_embedding_variable_invalid_shape(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = constant_op.constant([-1, -1], dtypes.int64)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      table = de.get_variable("t110",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val,
                              dim=2)

      # Shape [6] instead of [3, 2]
      values = constant_op.constant([0, 1, 2, 3, 4, 5], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Shape [2,3] instead of [3, 2]
      values = constant_op.constant([[0, 1, 2], [3, 4, 5]], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Shape [2, 2] instead of [3, 2]
      values = constant_op.constant([[0, 1], [2, 3]], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Shape [3, 1] instead of [3, 2]
      values = constant_op.constant([[0], [2], [4]], dtypes.int32)
      with self.assertRaisesOpError("Expected shape"):
        self.evaluate(table.upsert(keys, values))

      # Valid Insert
      values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int32)
      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

  def test_dynamic_embedding_variable_duplicate_insert(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2, 2], dtypes.int64)
      values = constant_op.constant([[0.], [1.], [2.], [3.]], dtypes.float32)
      table = de.get_variable("t130",
                              dtypes.int64,
                              dtypes.float32,
                              initializer=default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_keys = constant_op.constant([0, 1, 2], dtypes.int64)
      output = table.lookup(input_keys)

      result = self.evaluate(output)
      self.assertTrue(list(result) in [[[0.], [1.], [3.]], [[0.], [1.], [2.]]])

  def test_dynamic_embedding_variable_find_high_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.get_variable("t140",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_keys = constant_op.constant([[0, 1], [2, 4]], dtypes.int64)
      output = table.lookup(input_keys)
      self.assertAllEqual([2, 2, 1], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual([[[0], [1]], [[2], [-1]]], result)

  def test_dynamic_embedding_variable_insert_low_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      values = constant_op.constant([[[0], [1]], [[2], [3]]], dtypes.int32)
      table = de.get_variable("t150",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [1], [3], [-1]], result)

  def test_dynamic_embedding_variable_remove_low_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      values = constant_op.constant([[[0], [1]], [[2], [3]]], dtypes.int32)
      table = de.get_variable("t160",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_keys = constant_op.constant([1, 4], dtypes.int64)
      self.evaluate(table.remove(remove_keys))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3, 4], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [-1], [3], [-1]], result)

  def test_dynamic_embedding_variable_insert_high_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant([-1, -1, -1], dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int32)
      table = de.get_variable("t170",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val,
                              dim=3)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 1], [3, 4]], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual(
        [[[0, 1, 2], [2, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]], result)

  def test_dynamic_embedding_variable_remove_high_rank(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant([-1, -1, -1], dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                    dtypes.int32)
      table = de.get_variable("t180",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val,
                              dim=3)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 3]], dtypes.int64)
      self.evaluate(table.remove(remove_keys))
      self.assertAllEqual(2, self.evaluate(table.size()))

      remove_keys = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
      output = table.lookup(remove_keys)
      self.assertAllEqual([2, 2, 3], output.get_shape())

      result = self.evaluate(output)
      self.assertAllEqual(
        [[[-1, -1, -1], [2, 3, 4]], [[4, 5, 6], [-1, -1, -1]]], result)

  def test_dynamic_embedding_variables(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)

      table1 = de.get_variable("t191",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)
      table2 = de.get_variable("t192",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)
      table3 = de.get_variable("t193",
                               dtypes.int64,
                               dtypes.int32,
                               initializer=default_val)
      self.evaluate(table1.upsert(keys, values))
      self.evaluate(table2.upsert(keys, values))
      self.evaluate(table3.upsert(keys, values))

      self.assertAllEqual(3, self.evaluate(table1.size()))
      self.assertAllEqual(3, self.evaluate(table2.size()))
      self.assertAllEqual(3, self.evaluate(table3.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output1 = table1.lookup(remove_keys)
      output2 = table2.lookup(remove_keys)
      output3 = table3.lookup(remove_keys)

      out1, out2, out3 = self.evaluate([output1, output2, output3])
      self.assertAllEqual([[0], [1], [-1]], out1)
      self.assertAllEqual([[0], [1], [-1]], out2)
      self.assertAllEqual([[0], [1], [-1]], out3)

  def test_dynamic_embedding_variable_with_tensor_default(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      default_val = constant_op.constant(-1, dtypes.int32)
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.get_variable("t200",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllEqual([[0], [1], [-1]], result)

  def test_signature_mismatch(self):
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with self.session(config=config, use_gpu=test_util.is_gpu_available()):
      default_val = -1
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0], [1], [2]], dtypes.int32)
      table = de.get_variable("t210",
                              dtypes.int64,
                              dtypes.int32,
                              initializer=default_val)

      # upsert with keys of the wrong type
      with self.assertRaises(ValueError):
        self.evaluate(
          table.upsert(constant_op.constant([4., 5., 6.], dtypes.float32),
                       values))

      # upsert with values of the wrong type
      with self.assertRaises(ValueError):
        self.evaluate(table.upsert(keys, constant_op.constant(["a", "b", "c"])))

      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys_ref = variables.Variable(0, dtype=dtypes.int64)
      input_int64_ref = variables.Variable([-1], dtype=dtypes.int32)
      self.evaluate(variables.global_variables_initializer())

      # Ref types do not produce an upsert signature mismatch.
      self.evaluate(table.upsert(remove_keys_ref, input_int64_ref))
      self.assertAllEqual(3, self.evaluate(table.size()))

      # Ref types do not produce a lookup signature mismatch.
      self.assertEqual([-1], self.evaluate(table.lookup(remove_keys_ref)))

      # lookup with keys of the wrong type
      remove_keys = constant_op.constant([1, 2, 3], dtypes.int32)
      with self.assertRaises(ValueError):
        self.evaluate(table.lookup(remove_keys))

  def test_dynamic_embedding_variable_int_float(self):
    with self.session(config=default_config,
                      use_gpu=test_util.is_gpu_available()):
      default_val = -1.0
      keys = constant_op.constant([3, 7, 0], dtypes.int64)
      values = constant_op.constant([[7.5], [-1.2], [9.9]], dtypes.float32)
      table = de.get_variable("t220",
                              dtypes.int64,
                              dtypes.float32,
                              initializer=default_val)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([7, 0, 11], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertAllClose([[-1.2], [9.9], [default_val]], result)

  def test_dynamic_embedding_variable_with_random_init(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      keys = constant_op.constant([0, 1, 2], dtypes.int64)
      values = constant_op.constant([[0.], [1.], [2.]], dtypes.float32)
      default_val = init_ops.random_uniform_initializer()
      table = de.get_variable("t230",
                              dtypes.int64,
                              dtypes.float32,
                              initializer=default_val)

      self.evaluate(table.upsert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      remove_keys = constant_op.constant([0, 1, 3], dtypes.int64)
      output = table.lookup(remove_keys)

      result = self.evaluate(output)
      self.assertNotEqual([-1.], result[2])


@test_util.deprecated_graph_mode_only
class EmbeddingLookupTest(test.TestCase):

  def test_simple_sharded(self):
    embeddings = de.get_variable("t300",
                                 dtypes.int64,
                                 dtypes.float32,
                                 devices=_get_devices() * 2,
                                 initializer=2.0)

    ids = constant_op.constant([0, 1, 2, 3, 4], dtype=dtypes.int64)
    embedding, trainable = de.embedding_lookup(embeddings,
                                               ids,
                                               max_norm=1.0,
                                               return_trainable=True)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      self.assertAllClose(embedding.eval(), [
        [1.0],
      ] * 5)
      self.evaluate(trainable.update_op())
      self.assertAllEqual(embeddings.size().eval(), 5)
      self.assertAllEqual(embeddings.size(0).eval(), 3)
      self.assertAllEqual(embeddings.size(1).eval(), 2)

  def test_max_norm(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embeddings = de.get_variable("t310",
                                   dtypes.int64,
                                   dtypes.float32,
                                   initializer=2.0)

      ids = constant_op.constant([0], dtype=dtypes.int64)
      embedding = de.embedding_lookup(embeddings, ids, max_norm=1.0)
      self.assertAllEqual(embedding.eval(), [[1.0]])

  def test_max_norm_nontrivial(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embeddings = de.get_variable("t320",
                                   dtypes.int64,
                                   dtypes.float32,
                                   initializer=2.0,
                                   dim=2)
      fake_values = constant_op.constant([[2.0, 4.0], [3.0, 1.0]])
      ids = constant_op.constant([0, 1], dtype=dtypes.int64)
      self.evaluate(embeddings.upsert(ids, fake_values))
      embedding_no_norm = de.embedding_lookup(embeddings, ids)
      embedding = de.embedding_lookup(embeddings, ids, max_norm=2.0)
      norms = math_ops.sqrt(
        math_ops.reduce_sum(embedding_no_norm * embedding_no_norm, axis=1))
      normalized = embedding_no_norm / array_ops.stack([norms, norms], axis=1)
      self.assertAllEqual(embedding.eval(), 2 * self.evaluate(normalized))

  def test_sharded_custom_partitioner_int32_ids(self):

    def _partition_fn(keys, shard_num):
      return math_ops.cast(keys % 2, dtype=dtypes.int32)

    embeddings = de.get_variable("t330",
                                 dtypes.int64,
                                 dtypes.float32,
                                 partitioner=_partition_fn,
                                 devices=_get_devices() * 3,
                                 initializer=2.0)

    ids = constant_op.constant([0, 1, 2, 3, 4], dtype=dtypes.int64)
    vals = constant_op.constant([[0.], [1.], [2.], [3.], [4.]],
                                dtype=dtypes.float32)
    ids_test = constant_op.constant([1, 3, 2, 3, 0], dtype=dtypes.int64)
    embedding = de.embedding_lookup(embeddings, ids_test)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      self.evaluate(embeddings.upsert(ids, vals))
      self.assertAllClose(embedding.eval(), [[1.], [3.], [2.], [3.], [0.]])
      self.assertAllEqual([5, 1], embedding.eval().shape)
      self.assertAllEqual(3, embeddings.size(0).eval())
      self.assertAllEqual(2, embeddings.size(1).eval())
      self.assertAllEqual(0, embeddings.size(2).eval())

  def test_sharded_multi_lookup_on_one_variable(self):
    embeddings = de.get_variable("t340",
                                 dtypes.int64,
                                 dtypes.float32,
                                 devices=_get_devices() * 3,
                                 initializer=2.0)

    ids = constant_op.constant([0, 1, 2, 3, 4], dtype=dtypes.int64)
    vals = constant_op.constant([[0.], [1.], [2.], [3.], [4.]],
                                dtype=dtypes.float32)
    new_vals = constant_op.constant([[10.], [11.], [12.], [13.], [14.]],
                                    dtype=dtypes.float32)

    ids0 = constant_op.constant([1, 3, 2], dtype=dtypes.int64)
    ids1 = constant_op.constant([3, 4], dtype=dtypes.int64)

    embedding0 = de.embedding_lookup(embeddings, ids0)
    embedding1 = de.embedding_lookup(embeddings, ids1)

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      self.evaluate(embeddings.upsert(ids, vals))
      self.assertAllClose(embedding0.eval(), [[1.], [3.], [2.]])
      self.assertAllEqual([3, 1], embedding0.eval().shape)
      self.assertAllClose(embedding1.eval(), [[3.], [4.]])
      self.assertAllEqual([2, 1], embedding1.eval().shape)
      self.evaluate(embeddings.upsert(ids, new_vals))
      self.assertAllClose(embedding1.eval(), [[13.], [14.]])
      self.assertAllEqual([2, 1], embedding1.eval().shape)

  def test_higher_rank(self):
    np.random.seed(8)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      for dim in [1, 10]:
        for ids_shape in [[3, 2], [4, 3], [4, 3, 10]]:
          with variable_scope.variable_scope("test_higher_rank", reuse=True):
            params = de.get_variable("t350-" + str(dim),
                                     dtypes.int64,
                                     dtypes.float32,
                                     initializer=2.0,
                                     dim=dim)
            ids = np.random.randint(2 ** 31,
                                    size=np.prod(ids_shape),
                                    dtype=np.int).reshape(ids_shape)
            ids = constant_op.constant(ids, dtype=dtypes.int64)
            simple = params.lookup(ids)
            self.evaluate(params.upsert(ids, simple))

            embedding = de.embedding_lookup(params, ids)
            self.assertAllEqual(simple.eval(), embedding.eval())
            self.assertAllEqual(ids_shape + [dim], embedding.eval().shape)

  def test_static_shape_checking(self):
    np.random.seed(8)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      for dim in [1, 10]:
        for ids_shape in [[3, 2], [4, 3], [4, 3, 10]]:
          with variable_scope.variable_scope("test_static_shape_checking" +
                                             str(dim),
                                             reuse=variable_scope.AUTO_REUSE):
            params = de.get_variable("test_static_shape_checking-" + str(dim),
                                     dtypes.int64,
                                     dtypes.float32,
                                     initializer=2.0,
                                     dim=dim)
            params_nn = variable_scope.get_variable("n",
                                                    shape=[100, dim],
                                                    use_resource=False)
            ids = np.random.randint(2 ** 31,
                                    size=np.prod(ids_shape),
                                    dtype=np.int).reshape(ids_shape)
            ids = constant_op.constant(ids, dtype=dtypes.int64)

            embedding_test = de.embedding_lookup(params, ids)
            embedding_base = embedding_ops.embedding_lookup(params_nn, ids)
            self.assertAllEqual(embedding_test.shape, embedding_base.shape)

  def test_dynamic_shape_checking(self):
    np.random.seed(8)
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      for dim in [1, 10]:
        for ids_shape in [None, [-1, 1], [1, -1, 1], [-1, 1, 1]]:
          with variable_scope.variable_scope("test_static_shape_checking" +
                                             str(dim),
                                             reuse=variable_scope.AUTO_REUSE):
            params = de.get_variable("test_static_shape_checking-" + str(dim),
                                     dtypes.int64,
                                     dtypes.float32,
                                     initializer=2.0,
                                     dim=dim)
            params_nn = variable_scope.get_variable("n",
                                                    shape=[100, dim],
                                                    use_resource=False)
            ids = script_ops.py_func(_create_dynamic_shape_tensor(min_val=0,
                                                                  max_val=100),
                                     inp=[],
                                     Tout=dtypes.int64,
                                     stateful=True)
            if ids_shape is not None:
              ids = array_ops.reshape(ids, ids_shape)

            embedding_test = de.embedding_lookup(params, ids)
            embedding_base = embedding_ops.embedding_lookup(params_nn, ids)

            # check static shape
            if ids_shape is None:
              # ids with unknown shape
              self.assertTrue(embedding_test.shape == embedding_base.shape)
            else:
              # ids with no fully-defined shape.
              self.assertAllEqual(embedding_test.shape.as_list(),
                                  embedding_base.shape.as_list())

            self.evaluate(variables.global_variables_initializer())

            # check static shape
            for _ in range(10):
              embedding_test_val, embedding_base_val = self.evaluate(
                [embedding_test, embedding_base])
              self.assertAllEqual(embedding_test_val.shape,
                                  embedding_base_val.shape)

  def test_scope_reuse_embedding_lookup(self):
    ids = constant_op.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               dtype=dtypes.int64)
    with variable_scope.variable_scope('test', reuse=variable_scope.AUTO_REUSE):
      p1 = de.get_variable(name="p1")
      with variable_scope.variable_scope('q'):
        _, t1 = de.embedding_lookup(p1, ids, name="emb", return_trainable=True)

    with variable_scope.variable_scope('test', reuse=variable_scope.AUTO_REUSE):
      p1_reuse = de.get_variable(name="p1")
      p2 = de.get_variable(name="p2")
      with variable_scope.variable_scope('q'):
        _, t2 = de.embedding_lookup(p2, ids, name="emb", return_trainable=True)

    self.assertAllEqual(p1.name, "test/p1")
    self.assertAllEqual(p2.name, "test/p2")
    self.assertAllEqual(p1, p1_reuse)
    self.assertEqual(t1.name, "test/q/emb/TrainableWrapper:0")
    self.assertEqual(t2.name, "test/q/emb/TrainableWrapper_1:0")
    self.assertAllEqual(p1._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p1_reuse._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p2._tables[0].name, "test_p2_mht_1of1")

  def test_scope_reuse_sparse_embedding_lookup(self):
    indices = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [1, 0, 0], [1, 1, 0],
               [1, 1, 1]]
    ids = [0, 1, -1, -1, 2, 0, 1]
    shape = [2, 3, 4]

    sparse_ids = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(ids, dtypes.int64),
      constant_op.constant(shape, dtypes.int64))

    with variable_scope.variable_scope('test', reuse=variable_scope.AUTO_REUSE):
      p1 = de.get_variable(name="p1")
      with variable_scope.variable_scope('q'):
        _, t1 = de.embedding_lookup_sparse(p1,
                                           sparse_ids,
                                           None,
                                           name="sp_emb",
                                           return_trainable=True)

    with variable_scope.variable_scope('test', reuse=variable_scope.AUTO_REUSE):
      p1_reuse = de.get_variable(name="p1")
      p2 = de.get_variable(name="p2")
      with variable_scope.variable_scope('q'):
        _, t2 = de.embedding_lookup_sparse(p2,
                                           sparse_ids,
                                           None,
                                           name="sp_emb",
                                           return_trainable=True)

    self.assertAllEqual(p1.name, "test/p1")
    self.assertAllEqual(p2.name, "test/p2")
    self.assertAllEqual(p1, p1_reuse)
    self.assertEqual(t1.name,
                     "test/q/sp_emb/embedding_lookup/TrainableWrapper:0")
    self.assertEqual(t2.name,
                     "test/q/sp_emb/embedding_lookup/TrainableWrapper_1:0")
    self.assertAllEqual(p1._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p1_reuse._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p2._tables[0].name, "test_p2_mht_1of1")

  def test_scope_reuse_safe_sparse_embedding_lookup(self):
    indices = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [1, 0, 0], [1, 1, 0],
               [1, 1, 1]]
    ids = [0, 1, -1, -1, 2, 0, 1]
    shape = [2, 3, 4]

    sparse_ids = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(ids, dtypes.int64),
      constant_op.constant(shape, dtypes.int64))

    with variable_scope.variable_scope('test', reuse=variable_scope.AUTO_REUSE):
      p1 = de.get_variable(name="p1")
      with variable_scope.variable_scope('q'):
        _, t1 = de.safe_embedding_lookup_sparse(p1,
                                                sparse_ids,
                                                None,
                                                name="safe_sp_emb",
                                                return_trainable=True)

    with variable_scope.variable_scope('test', reuse=variable_scope.AUTO_REUSE):
      p1_reuse = de.get_variable(name="p1")
      p2 = de.get_variable(name="p2")
      with variable_scope.variable_scope('q'):
        _, t2 = de.safe_embedding_lookup_sparse(p2,
                                                sparse_ids,
                                                None,
                                                name="safe_sp_emb",
                                                return_trainable=True)

    self.assertAllEqual(p1.name, "test/p1")
    self.assertAllEqual(p2.name, "test/p2")
    self.assertAllEqual(p1, p1_reuse)
    self.assertEqual(
      t1.name,
      "test/q/safe_sp_emb/embedding_lookup_sparse/embedding_lookup/TrainableWrapper:0"
    )
    self.assertEqual(
      t2.name,
      "test/q/safe_sp_emb/embedding_lookup_sparse/embedding_lookup/TrainableWrapper_1:0"
    )
    self.assertAllEqual(p1._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p1_reuse._tables[0].name, "test_p1_mht_1of1")
    self.assertAllEqual(p2._tables[0].name, "test_p2_mht_1of1")

  def test_treated_as_worker_op_by_device_setter(self):
    num_ps_tasks = 2
    with ops.device('/job:worker/task:0'):
      ids = constant_op.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                 dtype=dtypes.int64)
    setter = device_setter.replica_device_setter(ps_tasks=num_ps_tasks,
                                                 ps_device='/job:ps',
                                                 worker_device='/job:worker')
    with ops.device(setter):
      p1 = de.get_variable(name="p1",
                           devices=['/job:ps/task:0', '/job:ps/task:1'])
      t1 = de.embedding_lookup(p1, ids, name="emb")
    self.assertTrue("/job:ps/task:0" in p1._tables[0].resource_handle.device)
    self.assertTrue("/job:ps/task:1" in p1._tables[1].resource_handle.device)

  def test_embedding_lookup_sparse_with_initializer(self):
    id = 0
    embed_dim = 8
    elements_num = 262144
    for initializer, target_mean, target_stddev in [
      (init_ops.random_normal_initializer(0.0, 0.001), 0.0, 0.001),
      (init_ops.truncated_normal_initializer(0.0, 0.001), 0.0, 0.00088),
      (keras_init_ops.RandomNormalV2(mean=0.0, stddev=0.001), 0.0, 0.001),
    ]:
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()):
        id += 1
        embedding_weights = de.get_variable('emb-init-bugfix-' + str(id),
                                            key_dtype=dtypes.int64,
                                            value_dtype=dtypes.float32,
                                            devices=_get_devices() * 3,
                                            initializer=initializer,
                                            dim=embed_dim)

        ids = np.random.randint(-0x7FFFFFFFFFFFFFFF,
                                0x7FFFFFFFFFFFFFFF,
                                elements_num,
                                dtype=np.int64)
        ids = np.unique(ids)
        ids = constant_op.constant(ids, dtypes.int64)
        vals_op = (de.embedding_lookup(embedding_weights, ids, None).eval())

        mean = self.evaluate(math_ops.reduce_mean(vals_op))
        stddev = self.evaluate(math_ops.reduce_std(vals_op))
        rtol = 2e-5
        atol = rtol
        self.assertTrue(not (list(vals_op[0]) == list(vals_op[1])))
        self.assertAllClose(target_mean, mean, rtol, atol)
        self.assertAllClose(target_stddev, stddev, rtol, atol)


@test_util.deprecated_graph_mode_only
class EmbeddingLookupSparseTest(test.TestCase):

  def _random_ids_and_weights(self, batch_size, vocab_size, k_type, d_type):
    max_val_per_entry = 6
    vals_per_batch_entry = np.random.randint(1,
                                             max_val_per_entry,
                                             size=batch_size)
    num_vals = np.sum(vals_per_batch_entry)

    ids = np.random.randint(vocab_size, size=num_vals)
    ids = ids.astype(_type_converter(k_type))
    weights = 1 + np.random.rand(num_vals)
    weights = weights.astype(_type_converter(d_type))

    indices = []
    for batch_entry, num_val in enumerate(vals_per_batch_entry):
      for val_index in range(num_val):
        indices.append([batch_entry, val_index])

    shape = [batch_size, max_val_per_entry]

    sp_ids = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(ids, k_type),
      constant_op.constant(shape, dtypes.int64))
    sp_weights = sparse_tensor.SparseTensor(
      constant_op.constant(indices, dtypes.int64),
      constant_op.constant(weights, d_type),
      constant_op.constant(shape, dtypes.int64))

    return sp_ids, sp_weights, ids, weights, vals_per_batch_entry

  def _group_by_batch_entry(self, vals, vals_per_batch_entry):
    grouped_vals = []
    index = 0
    for num_val in vals_per_batch_entry:
      grouped_vals.append(list(vals[index:(index + num_val)]))
      index += num_val
    return grouped_vals

  def test_embedding_lookup_sparse(self):

    var_id = 0
    for num_shards, initial_mode, combiner, \
        k_dtype, d_dtype, ignore_weights, dim in itertools.product(
      [1, 3], ['constant', 'random'], ["sum", "mean", "sqrtn", ],
      [dtypes.int64],
      [dtypes.float32, ],
      [True, False],
      [1, 5]):
      vocab_size = 2 ** 31 if k_dtype == dtypes.int32 else 2 ** 63
      batch_size = 5

      sp_ids, sp_weights, ids, weights, vals_per_batch_entry = (
        self._random_ids_and_weights(batch_size, vocab_size, k_dtype,
                                     d_dtype))

      grouped_ids = self._group_by_batch_entry(ids, vals_per_batch_entry)
      grouped_weights = self._group_by_batch_entry(weights,
                                                   vals_per_batch_entry)
      grouped_ignored_weights = self._group_by_batch_entry(
        np.ones(np.sum(vals_per_batch_entry)), vals_per_batch_entry)

      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        var_id += 1
        params = de.get_variable('t1000-' + str(var_id),
                                 key_dtype=k_dtype,
                                 value_dtype=d_dtype,
                                 devices=_get_devices() * num_shards,
                                 initializer=1.,
                                 dim=dim)
        random_init = params.lookup(ids)
        init_op = params.upsert(ids, random_init)
        self.evaluate(init_op)
        np_params = random_init.eval()
        grouped_params = self._group_by_batch_entry(np_params,
                                                    vals_per_batch_entry)
        embedding_sum = de.embedding_lookup_sparse(
          params,
          sp_ids,
          None if ignore_weights else sp_weights,
          combiner=combiner)
        self.assertEqual(embedding_sum.dtype, d_dtype)

        tf_embedding_sum = embedding_sum.eval()

        np_embedding_sum, np_weight_sum, np_weight_sq_sum = embedding_result(
          grouped_params,
          grouped_ids,
          weight_vals=grouped_ignored_weights
          if ignore_weights else grouped_weights)
        if combiner == "mean":
          np_embedding_sum /= np.reshape(np_weight_sum, (batch_size, 1))
        if combiner == "sqrtn":
          np_embedding_sum /= np.reshape(np.sqrt(np_weight_sq_sum),
                                         (batch_size, 1))

        rtol = 1e-6
        atol = rtol
        self.assertAllClose(np_embedding_sum, tf_embedding_sum, rtol, atol)

  def test_embedding_lookup_sparse_shape_checking(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embed_dim = 4
      embedding_weights_nn = variable_scope.get_variable("n",
                                                         shape=[100, embed_dim],
                                                         use_resource=False)
      embedding_weights_de = _random_weights(embed_dim=4)
      sparse_ids, _ = ids_and_weights_3d(embed_dim=embed_dim)

      embedding_lookup_base = embedding_ops.embedding_lookup_sparse(
        embedding_weights_nn, sparse_ids, None)
      embedding_lookup_test = de.embedding_lookup_sparse(
        embedding_weights_de, sparse_ids, None)
      self.assertTrue(embedding_lookup_base.get_shape().as_list() ==
                      embedding_lookup_test.get_shape().as_list())


@test_util.deprecated_graph_mode_only
class SafeEmbeddingLookupSparseTest(test.TestCase):

  def test_safe_embedding_lookup_sparse_return_zero_vector(self):
    with self.cached_session(use_gpu=test_util.is_gpu_available(),
                             config=default_config):
      dim = 4
      embedding_weights = _random_weights(embed_dim=dim)
      sparse_ids, sparse_weights = ids_and_weights_2d(embed_dim=dim)
      valid_ids = np.array([
        0,
        1,
        2,
        -1,
      ])

      # init
      embedding_weights_values = embedding_weights.lookup(valid_ids).eval()
      self.evaluate(
        embedding_weights.upsert(valid_ids, embedding_weights_values))

      # check
      embedding_lookup_result = \
        de.safe_embedding_lookup_sparse(embedding_weights, sparse_ids,
                                        sparse_weights).eval()
      self.assertAllClose(embedding_lookup_result, [
        (1.0 * embedding_weights_values[0] + 2.0 * embedding_weights_values[1]
         + 1.0 * embedding_weights_values[3]) / 4.0,
        embedding_weights_values[3] * 1.0, [0] * dim,
        embedding_weights_values[2], [0] * dim
      ])

  def test_safe_embedding_lookup_sparse_return_special_vector(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      dim = 4
      embedding_weights = _random_weights(embed_dim=dim)
      sparse_ids, sparse_weights = ids_and_weights_2d(embed_dim=dim)
      valid_ids = np.array([0, 1, 2, 3, -1])

      # init
      embedding_weights_values = embedding_weights.lookup(valid_ids).eval()
      self.evaluate(
        embedding_weights.upsert(valid_ids, embedding_weights_values))

      # check
      embedding_lookup_result = de.safe_embedding_lookup_sparse(
        embedding_weights, sparse_ids, sparse_weights, default_id=3).eval()

      self.assertAllClose(embedding_lookup_result, [
        (1.0 * embedding_weights_values[0] + 2.0 * embedding_weights_values[1]
         + 1.0 * embedding_weights_values[4]) / 4.0,
        embedding_weights_values[4], embedding_weights_values[3],
        embedding_weights_values[2], embedding_weights_values[3]
      ])

  def test_safe_embedding_lookup_sparse_no_weights(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      dim = 4
      embedding_weights = _random_weights(embed_dim=dim)
      sparse_ids, sparse_weights = ids_and_weights_2d(embed_dim=dim)
      valid_ids = np.array([0, 1, 2, -1])

      # init
      embedding_weights_values = embedding_weights.lookup(valid_ids).eval()
      self.evaluate(
        embedding_weights.upsert(valid_ids, embedding_weights_values))

      embedding_lookup_result = (de.safe_embedding_lookup_sparse(
        embedding_weights, sparse_ids, None).eval())

      self.assertAllClose(
        embedding_lookup_result,
        [(embedding_weights_values[0] + embedding_weights_values[1] +
          embedding_weights_values[3]) / 3.0, embedding_weights_values[3],
         [0] * 4, embedding_weights_values[2],
         (embedding_weights_values[0] + embedding_weights_values[1]) / 2.0])

  def test_safe_embedding_lookup_sparse_partitioned(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      dim = 4
      embedding_weights = _random_weights(embed_dim=dim, num_shards=3)
      sparse_ids, sparse_weights = ids_and_weights_2d(embed_dim=dim)
      valid_ids = np.array([0, 1, 2, -1])

      # init
      embedding_weights_values = embedding_weights.lookup(valid_ids).eval()
      self.evaluate(
        embedding_weights.upsert(valid_ids, embedding_weights_values))

      embedding_lookup_result = (de.safe_embedding_lookup_sparse(
        embedding_weights, sparse_ids, None).eval())

      self.assertAllClose(
        embedding_lookup_result,
        [(embedding_weights_values[0] + embedding_weights_values[1] +
          embedding_weights_values[3]) / 3.0, embedding_weights_values[3],
         [0] * 4, embedding_weights_values[2],
         (embedding_weights_values[0] + embedding_weights_values[1]) / 2.0])

  def test_safe_embedding_lookup_sparse_inconsistent_ids_type(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      def fn():
        embedding_weights = _random_weights(num_shards=3,
                                            key_dtype=dtypes.int32)
        sparse_ids, sparse_weights = ids_and_weights_2d()
        de.safe_embedding_lookup_sparse(embedding_weights, sparse_ids,
                                        sparse_weights)

      self.assertRaises(TypeError, fn)

  def test_safe_embedding_lookup_sparse_inconsistent_weights_type(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      def fn():
        embedding_weights = _random_weights(num_shards=3, key_dtype=dtypes.half)
        sparse_ids, sparse_weights = ids_and_weights_2d()
        de.safe_embedding_lookup_sparse(embedding_weights, sparse_ids,
                                        sparse_weights)

      self.assertRaises(TypeError, fn)

  def test_safe_embedding_lookup_sparse_3d_return_zero_vector(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embedding_weights = _random_weights()
      sparse_ids, sparse_weights = ids_and_weights_3d()
      valid_ids = np.array([0, 1, 2, -1])
      # init
      embedding_weights_values = embedding_weights.lookup(valid_ids).eval()
      self.evaluate(
        embedding_weights.upsert(valid_ids, embedding_weights_values))

      embedding_lookup_result = (de.safe_embedding_lookup_sparse(
        embedding_weights, sparse_ids, sparse_weights).eval())

      self.assertAllClose(embedding_lookup_result, [[
        (1.0 * embedding_weights_values[0] + 2.0 * embedding_weights_values[1]
         + 1.0 * embedding_weights_values[3]) / 4.0,
        embedding_weights_values[3], [0] * 4
      ], [embedding_weights_values[2], [0] * 4, [0] * 4]])

  def test_safe_embedding_lookup_sparse_3d_return_special_vector(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embedding_weights = _random_weights()
      sparse_ids, sparse_weights = ids_and_weights_3d()
      valid_ids = np.array([0, 1, 2, 3, -1])

      # init
      embedding_weights_values = embedding_weights.lookup(valid_ids).eval()
      self.evaluate(
        embedding_weights.upsert(valid_ids, embedding_weights_values))

      embedding_lookup_result = (de.safe_embedding_lookup_sparse(
        embedding_weights, sparse_ids, sparse_weights, default_id=3).eval())

      self.assertAllClose(
        embedding_lookup_result,
        [[(1.0 * embedding_weights_values[0] + 2.0 *
           embedding_weights_values[1] + 1.0 * embedding_weights_values[4]) /
          4.0, embedding_weights_values[4], embedding_weights_values[3]],
         [
           embedding_weights_values[2], embedding_weights_values[3],
           embedding_weights_values[3]
         ]])

  def test_safe_embedding_lookup_sparse_3d_no_weights(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embedding_weights = _random_weights()
      sparse_ids, _ = ids_and_weights_3d()
      valid_ids = np.array([0, 1, 2, -1])
      # init
      embedding_weights_values = embedding_weights.lookup(valid_ids).eval()
      self.evaluate(
        embedding_weights.upsert(valid_ids, embedding_weights_values))

      embedding_lookup_result = (de.safe_embedding_lookup_sparse(
        embedding_weights, sparse_ids, None).eval())

      self.assertAllClose(
        embedding_lookup_result,
        [[(embedding_weights_values[0] + embedding_weights_values[1] +
           embedding_weights_values[3]) / 3.0, embedding_weights_values[3],
          [0] * 4],
         [
           embedding_weights_values[2],
           (embedding_weights_values[0] + embedding_weights_values[1]) /
           2.0, [0] * 4
         ]])

  def test_safe_embedding_lookup_sparse_3d_partitioned(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embedding_weights = _random_weights(num_shards=3)
      sparse_ids, _ = ids_and_weights_3d()
      valid_ids = np.array([0, 1, 2, -1])

      # init
      embedding_weights_values = embedding_weights.lookup(valid_ids).eval()
      self.evaluate(
        embedding_weights.upsert(valid_ids, embedding_weights_values))

      embedding_lookup_result = (de.safe_embedding_lookup_sparse(
        embedding_weights, sparse_ids, None).eval())

      self.assertAllClose(
        embedding_lookup_result,
        [[(embedding_weights_values[0] + embedding_weights_values[1] +
           embedding_weights_values[3]) / 3.0, embedding_weights_values[3],
          [0] * 4],
         [
           embedding_weights_values[2],
           (embedding_weights_values[0] + embedding_weights_values[1]) /
           2.0, [0] * 4
         ]])

  def test_safe_embedding_lookup_sparse_with_initializer(self):
    id = 0
    embed_dim = 8
    dense_shape = np.array([64, 128, 32])
    total_space = 64 * 128 * 32
    elements_num = int(total_space * 0.50)
    for initializer, target_mean, target_stddev in [
      (init_ops.random_normal_initializer(0.0, 0.001), 0.0, 0.00029),
      (init_ops.truncated_normal_initializer(0.0, 0.001), 0.0, 0.00029),
      (keras_init_ops.RandomNormalV2(mean=0.0, stddev=0.001), 0.0, 0.00029),
    ]:
      with self.session(config=default_config,
                        use_gpu=test_util.is_gpu_available()):
        id += 1
        embedding_weights = de.get_variable('safe-init-bugfix-' + str(id),
                                            key_dtype=dtypes.int64,
                                            value_dtype=dtypes.float32,
                                            devices=_get_devices() * 3,
                                            initializer=initializer,
                                            dim=embed_dim)

        indices_1d = np.random.randint(0, total_space, elements_num)
        indices_1d = np.unique(indices_1d)
        indices_1d.sort()
        indices_3d = []
        for _i in range(indices_1d.size):
          a_indice = []
          quotient = int(indices_1d[_i] / (128 * 32))
          remainder = indices_1d[_i] % (128 * 32)
          a_indice.append(quotient)
          quotient = int(remainder / 32)
          remainder = remainder % 32
          a_indice.extend([quotient, remainder])
          indices_3d.extend([a_indice])
        indices_3d = np.array(indices_3d)

        ids = np.random.randint(-0x7FFFFFFFFFFFFFFF,
                                0x7FFFFFFFFFFFFFFF,
                                indices_1d.size,
                                dtype=np.int64)

        sparse_ids = sparse_tensor.SparseTensor(
          constant_op.constant(indices_3d, dtypes.int64),
          constant_op.constant(ids, dtypes.int64),
          constant_op.constant(dense_shape, dtypes.int64))
        vals_op = (de.safe_embedding_lookup_sparse(embedding_weights,
                                                   sparse_ids,
                                                   None,
                                                   combiner="mean").eval())

        mean = self.evaluate(math_ops.reduce_mean(vals_op))
        stddev = self.evaluate(math_ops.reduce_std(vals_op))
        rtol = 2e-4
        atol = rtol
        self.assertTrue(not (vals_op[0][0][0] == vals_op[0][0][1]))
        self.assertAllClose(target_mean, mean, rtol, atol)
        self.assertAllClose(target_stddev, stddev, rtol, atol)

  def test_safe_embedding_lookup_sparse_shape_checking(self):
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      embed_dim = 4
      embedding_weights_nn = variable_scope.get_variable("n",
                                                         shape=[100, embed_dim],
                                                         use_resource=False)
      embedding_weights_de = _random_weights(embed_dim=4)
      sparse_ids, _ = ids_and_weights_3d(embed_dim=embed_dim)

      embedding_lookup_base = embedding_ops.safe_embedding_lookup_sparse(
        embedding_weights_nn, sparse_ids, None)
      embedding_lookup_test = de.safe_embedding_lookup_sparse(
        embedding_weights_de, sparse_ids, None)
      self.assertAllEqual(embedding_lookup_base.shape,
                          embedding_lookup_test.shape)
      self.assertAllEqual(embedding_lookup_base.get_shape(),
                          embedding_lookup_test.get_shape())


if __name__ == '__main__':
  os.environ['TF_HASHTABLE_INIT_SIZE'] = '100000'
  test.main()
