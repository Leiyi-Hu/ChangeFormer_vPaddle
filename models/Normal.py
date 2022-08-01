#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import math
from paddle import framework
from paddle.fluid import core
from paddle.fluid.framework import _non_static_mode, in_dygraph_mode, _in_legacy_dygraph, default_main_program, \
    _current_expected_place
import numpy as np
from paddle.fluid.core import VarDesc
from paddle.fluid import unique_name
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from paddle import _C_ops

__all__ = [
    'XavierNormal','Xavier', 'XavierInitializer','set_global_initializer'
]

_global_weight_initializer_ = None
_global_bias_initializer_ = None


class Initializer(object):
    """Base class for variable initializers

    Defines the common interface of variable initializers.
    They add operations to the init program that are used
    to initialize variables. Users should not use this class
    directly, but need to use one of its implementations.
    """

    def __init__(self):
        pass

    def __call__(self, param, block=None):
        """Add corresponding initialization operations to the network
        """
        raise NotImplementedError()

    def _check_block(self, block):
        if block is None:
            block = default_main_program().global_block()

        return block

    def _compute_fans(self, var):
        """Compute the fan_in and the fan_out for layers

        This method computes the fan_in and the fan_out
        for neural network layers, if not specified. It is
        not possible to perfectly estimate fan_in and fan_out.
        This method will estimate it correctly for matrix multiply and
        convolutions.

        Args:
            var: variable for which fan_in and fan_out have to be computed

        Returns:
            tuple of two integers (fan_in, fan_out)
        """
        shape = var.shape
        if not shape or len(shape) == 0:
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            # This is the case for simple matrix multiply
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assume this to be a convolutional kernel
            # In PaddlePaddle, the shape of the kernel is like:
            # [num_filters, num_filter_channels, ...] where the remaining
            # dimensions are the filter_size
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size

        return (fan_in, fan_out)

class XavierInitializer(Initializer):
    r"""
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.

    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where

    .. math::

        x = \sqrt{\\frac{6.0}{fan\_in + fan\_out}}

    In case of Normal distribution, the mean is 0 and the standard deviation
    is

    .. math::

        \sqrt{\\frac{2.0}{fan\_in + fan\_out}}


    Args:
        uniform (bool,default True): whether to use uniform ,if False use normal distribution
        fan_in (float,default None): fan_in for Xavier initialization. If None, it is
                inferred from the variable.
        fan_out (float,default None): fan_out for Xavier initialization. If None, it is
                 inferred from the variable.
        seed (int): random seed

    Note:
        It is recommended to set fan_in and fan_out to None for most cases.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            queries = fluid.data(name='x', shape=[None,1], dtype='float32')
            fc = fluid.layers.fc(
                input=queries, size=10,
                param_attr=fluid.initializer.Xavier(uniform=False))

    """

    def __init__(self,gain = 1.0, uniform=True, fan_in=None, fan_out=None, seed=0):
        assert uniform is not None
        assert seed is not None
        super(XavierInitializer, self).__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = seed
        self._gain = gain

    def __call__(self, var, block=None):
        """Initialize the input tensor with Xavier initialization.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out",
                                 ["uint16", "float16", "float32", "float64"],
                                 "xavier_init")

        f_in, f_out = self._compute_fans(var)

        # If fan_in and fan_out are passed, use them
        fan_in = f_in if self._fan_in is None else self._fan_in
        fan_out = f_out if self._fan_out is None else self._fan_out

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16 or (
                var.dtype == VarDesc.VarType.BF16 and not self._uniform):
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['xavier_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if framework._non_static_mode():
            if self._uniform:
                limit = np.sqrt(6.0 / float(fan_in + fan_out))
                out_var = _C_ops.uniform_random('shape', out_var.shape, 'min',
                                                -limit, 'max', limit, 'seed',
                                                self._seed, 'dtype', out_dtype)
            else:
                std = self._gain * math.sqrt(2.0 / float(fan_in + fan_out))

                if in_dygraph_mode():
                    place = _current_expected_place()
                    out_var = _C_ops.final_state_gaussian_random(
                        out_var.shape, 0.0, std, self._seed, out_dtype, place)
                else:
                    out_var = _C_ops.gaussian_random(
                        'shape', out_var.shape, 'dtype', out_dtype, 'mean', 0.0,
                        'std', std, 'seed', self._seed)

            if var.dtype == VarDesc.VarType.FP16 or (
                    var.dtype == VarDesc.VarType.BF16 and not self._uniform):
                var_tmp = _C_ops.cast(out_var, 'in_dtype', out_var.dtype,
                                      'out_dtype', var.dtype)
                var_tmp._share_underline_tensor_to(var)
            else:
                out_var._share_underline_tensor_to(var)
            return None
        else:
            if self._uniform:
                limit = np.sqrt(6.0 / float(fan_in + fan_out))
                op = block.append_op(
                    type="uniform_random",
                    inputs={},
                    outputs={"Out": out_var},
                    attrs={
                        "shape": out_var.shape,
                        "dtype": out_dtype,
                        "min": -limit,
                        "max": limit,
                        "seed": self._seed
                    },
                    stop_gradient=True)
            else:
                std = self._gain * np.sqrt(2.0 / float(fan_in + fan_out))
                op = block.append_op(
                    type="gaussian_random",
                    outputs={"Out": out_var},
                    attrs={
                        "shape": out_var.shape,
                        "dtype": out_dtype,
                        "mean": 0.0,
                        "std": std,
                        "seed": self._seed
                    },
                    stop_gradient=True)

            if var.dtype == VarDesc.VarType.FP16 or (
                    var.dtype == VarDesc.VarType.BF16 and not self._uniform):
                block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": var},
                    attrs={"in_dtype": out_var.dtype,
                           "out_dtype": var.dtype})

            var.op = op
            return op


def set_global_initializer(weight_init, bias_init=None):
    """
    This API is used to set up global model parameter initializer in framework.

    After this API is invoked, the global initializer will takes effect in subsequent code.

    The model parameters include ``weight`` and ``bias`` . In the framework, they correspond
    to ``paddle.ParamAttr`` , which is inherited from ``paddle.Tensor`` , and is a persistable Variable.
    This API only takes effect for model parameters, not for variables created through apis such as
    :ref:`api_fluid_layers_create_global_var` , :ref:`api_fluid_layers_create_tensor`.

    If the initializer is also set up by ``param_attr`` or ``bias_attr`` when creating a network layer,
    the global initializer setting here will not take effect because it has a lower priority.

    If you want to cancel the global initializer in framework, please set global initializer to ``None`` .

    Args:
        weight_init (Initializer): set the global initializer for ``weight`` of model parameters.
        bias_init (Initializer, optional): set the global initializer for ``bias`` of model parameters.
            Default: None.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle
            import paddle.nn as nn

            nn.initializer.set_global_initializer(nn.initializer.Uniform(), nn.initializer.Constant())
            x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)

            # The weight of conv1 is initialized by Uniform
            # The bias of conv1 is initialized by Constant
            conv1 = nn.Conv2D(4, 6, (3, 3))
            y_var1 = conv1(x_var)

            # If set param_attr/bias_attr too, global initializer will not take effect
            # The weight of conv2 is initialized by Xavier
            # The bias of conv2 is initialized by Normal
            conv2 = nn.Conv2D(4, 6, (3, 3),
                weight_attr=nn.initializer.XavierUniform(),
                bias_attr=nn.initializer.Normal())
            y_var2 = conv2(x_var)

            # Cancel the global initializer in framework, it will takes effect in subsequent code
            nn.initializer.set_global_initializer(None)
    """

    check_type(weight_init, 'weight_init', (Initializer, type(None)),
               'set_global_initializer')
    global _global_weight_initializer_
    _global_weight_initializer_ = weight_init

    check_type(bias_init, 'bias_init', (Initializer, type(None)),
               'set_global_initializer')
    global _global_bias_initializer_
    _global_bias_initializer_ = bias_init


def _global_weight_initializer():
    """
    Return the global weight initializer, The user doesn't need to use it.
    """
    return _global_weight_initializer_


def _global_bias_initializer():
    """
    Return the global weight initializer, The user doesn't need to use it.
    """
    return _global_bias_initializer_


def calculate_gain(nonlinearity, param=None):
    """
    Get the recommended ``gain`` value of some nonlinearity function. ``gain`` value can be used in some
    ``paddle.nn.initializer`` api to adjust the initialization value.

    Args:
        nonlinearity(str): name of nonlinearity activation function. If it is a linear function, such as:
            `linear/conv1d/conv2d/conv3d/conv1d_transpose/conv2d_transpose/conv3d_transpose` , 1.0 will be returned.
        param(bool|int|float, optional): optional parameter for somme nonlinearity function. Now, it only applies to
            'leaky_relu'. Default: None, it will be calculated as 0.01 in the formula.

    Returns:
        A float value, which is the recommended gain for this nonlinearity function.

    Examples:
        .. code-block:: python

            import paddle
            gain = paddle.nn.initializer.calculate_gain('tanh') # 5.0 / 3
            gain = paddle.nn.initializer.calculate_gain('leaky_relu', param=1.0) # 1.0 = math.sqrt(2.0 / (1+param^2))

    """
    if param is None:
        param = 0.01
    else:
        assert isinstance(param, (bool, int, float))
        param = float(param)
    recommended_gain = {
        'sigmoid': 1,
        'linear': 1,
        'conv1d': 1,
        'conv2d': 1,
        'conv3d': 1,
        'conv1d_transpose': 1,
        'conv2d_transpose': 1,
        'conv3d_transpose': 1,
        'tanh': 5.0 / 3,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + param ** 2)),
        'selu': 3.0 / 4
    }
    if nonlinearity in recommended_gain.keys():
        return recommended_gain[nonlinearity]
    else:
        raise ValueError("nonlinearity function {} is not suppported now.".
                         format(nonlinearity))


# We short the class name, since users will use the initializer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# hidden = fluid.layers.fc(...,
#                          param_attr=ParamAttr(fluid.initializer.Xavier()))
#
# It is no need to add an `Initializer` as the class suffix
Xavier = XavierInitializer

class XavierNormal(XavierInitializer):
    r"""
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio, using a normal distribution.

    The mean is 0 and the standard deviation is

    .. math::

        \sqrt{\frac{2.0}{fan\_in + fan\_out}}


    Args:
        fan_in (float, optional): fan_in for Xavier initialization, It is
                inferred from the tensor. The default value is None.
        fan_out (float, optional): fan_out for Xavier initialization, it is
                 inferred from the tensor. The default value is None.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A parameter initialized by Xavier weight, using a normal distribution.

    Examples:
        .. code-block:: python

            import paddle

            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.XavierNormal())
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.XavierNormal())
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            # inear.weight:  [[ 0.06910077 -0.18103665]
            #                 [-0.02546741 -1.0402188 ]]
            # linear.bias:  [-0.5012929   0.12418364]

            res = linear(data)
            # res:  [[[-0.4576595 -1.0970719]]
            #        [[-0.4576595 -1.0970719]]
            #        [[-0.4576595 -1.0970719]]]
    """

    def __init__(self, gain = 1.0 ,fan_in=None, fan_out=None, name=None):
        super(XavierNormal, self).__init__(
            gain = 1.0,uniform=False, fan_in=fan_in, fan_out=fan_out, seed=0)
