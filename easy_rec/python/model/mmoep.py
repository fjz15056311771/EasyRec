# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import tensorflow as tf

from easy_rec.python.layers import dnn
from easy_rec.python.layers import mmoe
from easy_rec.python.layers.keras.activation import activation_layer
from easy_rec.python.model.multi_task_model import MultiTaskModel
from easy_rec.python.protos.mmoep_pb2 import MMoEP as MMoEPConfig
from easy_rec.python.utils.tf_utils import add_elements_to_collection

if tf.__version__ >= '2.0':
  tf = tf.compat.v1

class GateNN(tf.keras.layers.Layer):

  def __init__(self,
               params,
               output_units=None,
               name='gate_nn',
               reuse=None,
               **kwargs):
    super(GateNN, self).__init__(name=name, **kwargs)
    output_dim = output_units if output_units is not None else params.output_dim
    hidden_dim = params.hidden_dim
    initializer = 'he_uniform'
    do_batch_norm = False
    activation = 'relu'
    dropout_rate = 0.0

    self._sub_layers = []
    dense = tf.keras.layers.Dense(
        units=hidden_dim,
        use_bias=not do_batch_norm,
        kernel_initializer=initializer,
        name=name)
    self._sub_layers.append(dense)

    if do_batch_norm:
      bn = tf.keras.layers.BatchNormalization(
          name='%s/bn' % name, trainable=True)
      self._sub_layers.append(bn)

    act_layer = activation_layer(activation)
    self._sub_layers.append(act_layer)

    if 0.0 < dropout_rate < 1.0:
      dropout = tf.keras.layers.Dropout(dropout_rate, name='%s/dropout' % name)
      self._sub_layers.append(dropout)
    elif dropout_rate >= 1.0:
      raise ValueError('invalid dropout_ratio: %.3f' % dropout_rate)

    dense = tf.keras.layers.Dense(
        units=output_dim,
        activation='sigmoid',
        use_bias=not do_batch_norm,
        kernel_initializer=initializer,
        name=name)
    self._sub_layers.append(dense)
    self._sub_layers.append(lambda x: x * 2)

  def call(self, x, training=None, **kwargs):
    """Performs the forward computation of the block."""
    for layer in self._sub_layers:
      cls = layer.__class__.__name__
      if cls in ('Dropout', 'BatchNormalization', 'Dice'):
        x = layer(x, training=training)
        if cls in ('BatchNormalization', 'Dice'):
          add_elements_to_collection(layer.updates, tf.GraphKeys.UPDATE_OPS)
      else:
        x = layer(x)
    return x

class MMoEP(MultiTaskModel):

  def __init__(self,
               model_config,
               feature_configs,
               features,
               labels=None,
               is_training=False):
    super(MMoEP, self).__init__(model_config, feature_configs, features, labels,
                               is_training)
    assert self._model_config.WhichOneof('model') == 'mmoep', \
        'invalid model config: %s' % self._model_config.WhichOneof('model')
    self._model_config = self._model_config.mmoep
    assert isinstance(self._model_config, MMoEPConfig)

    self.ppnet_config = self._model_config.ppnet
    self.full_gate_input = self.ppnet_config.full_gate_input
    gate_params = self.ppnet_config.gate_params
    l2_reg = float(self._model_config.l2_regularization)

    if self.has_backbone:
      self._features = self.backbone
    else:
      self._features_memorize, _ = self._input_layer(self._feature_dict, 'memorize')
      self._features_general, _ = self._input_layer(self._feature_dict, 'general')
    self._init_towers(self._model_config.task_towers)
    self.ppnet_layer = []
    self.tower_layer = {}
    units = list(self._model_config.task_towers[0].dnn.hidden_units)
    for i,num_units in enumerate(units):
      name = 'gate_%d' % i
      self.ppnet_layer.append(GateNN(gate_params, num_units, name))
    
    for i, task_tower_cfg in enumerate(self._model_config.task_towers):
      name = task_tower_cfg.tower_name
      layer_tmp = []
      for j,dnn_num in enumerate(list(task_tower_cfg.dnn.hidden_units)):
        act_layer = activation_layer(task_tower_cfg.dnn.activation)
        dense = tf.keras.layers.Dense(
            units=dnn_num,
            use_bias=True,
            kernel_initializer='he_uniform',
            kernel_regularizer=None,
            name=name)
        layer_tmp.append(dense)
        if task_tower_cfg.dnn.use_bn:
          bn = tf.keras.layers.BatchNormalization(
              name='%s/bn' % name, trainable=True)
          layer_tmp.append(bn)
        layer_tmp.append(act_layer)
        # dropout_rate = task_tower_cfg.dnn.dropout_ratio
        dropout_rate = 0.1
        if 0.0 < dropout_rate < 1.0:
          dropout = tf.keras.layers.Dropout(dropout_rate, name='%s/dropout' % name)
          layer_tmp.append(dropout)
        elif dropout_rate >= 1.0:
          raise ValueError('invalid dropout_ratio: %.3f' % dropout_rate)
        layer_tmp.append(self.ppnet_layer[j])
      self.tower_layer[name] = layer_tmp


  def build_predict_graph(self):
    if self._model_config.HasField('expert_dnn'):
      mmoe_layer = mmoe.MMOE(
          self._model_config.expert_dnn,
          l2_reg=self._l2_reg,
          num_task=self._task_num,
          num_expert=self._model_config.num_expert)
    else:
      # For backward compatibility with original mmoe layer config
      mmoe_layer = mmoe.MMOE([x.dnn for x in self._model_config.experts],
                             l2_reg=self._l2_reg,
                             num_task=self._task_num)
    task_input_list = mmoe_layer(self._features_general)

    tower_outputs = {}
    for index,(key,value) in enumerate(self.tower_layer.items()):
      tower_name = key
      x = task_input_list[index]
      gate_input = self._features_memorize
      if self.full_gate_input:
        with tf.name_scope('mmoep'):
          gate_input = tf.concat([tf.stop_gradient(self._features_general), gate_input], axis=-1)

      for layer in value:
        cls = layer.__class__.__name__
        if cls == 'GateNN':
          gate = layer(gate_input)
          x *= gate
        elif cls in ('Dropout', 'BatchNormalization', 'Dice'):
          x = layer(x, training=True)
          if cls in ('BatchNormalization', 'Dice'):
            add_elements_to_collection(layer.updates, tf.GraphKeys.UPDATE_OPS)
        else:
          x = layer(x)

      tower_output = tf.layers.dense(
          inputs=x,
          units=self._model_config.task_towers[index].num_class,
          kernel_regularizer=self._l2_reg,
          name='dnn_output_%d' % index)

      tower_outputs[tower_name] = tower_output
    self._add_to_prediction_dict(tower_outputs)
    return self._prediction_dict
