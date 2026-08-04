


def forward(self, L_x_seq_ : torch.Tensor, L_self_modules_layer_modules_features_modules_0_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_1_buffers_running_mean_ : torch.Tensor, L_self_modules_layer_modules_features_modules_1_buffers_running_var_ : torch.Tensor, L_self_modules_layer_modules_features_modules_1_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_1_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_4_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_5_buffers_running_mean_ : torch.Tensor, L_self_modules_layer_modules_features_modules_5_buffers_running_var_ : torch.Tensor, L_self_modules_layer_modules_features_modules_5_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_5_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_8_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_9_buffers_running_mean_ : torch.Tensor, L_self_modules_layer_modules_features_modules_9_buffers_running_var_ : torch.Tensor, L_self_modules_layer_modules_features_modules_9_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_9_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_11_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_12_buffers_running_mean_ : torch.Tensor, L_self_modules_layer_modules_features_modules_12_buffers_running_var_ : torch.Tensor, L_self_modules_layer_modules_features_modules_12_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_12_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_14_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_15_buffers_running_mean_ : torch.Tensor, L_self_modules_layer_modules_features_modules_15_buffers_running_var_ : torch.Tensor, L_self_modules_layer_modules_features_modules_15_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_features_modules_15_parameters_bias_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_classifier_modules_1_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_classifier_modules_4_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_layer_modules_classifier_modules_6_parameters_weight_ : torch.nn.parameter.Parameter):
    l_x_seq_ = L_x_seq_
    l_self_modules_layer_modules_features_modules_0_parameters_weight_ = L_self_modules_layer_modules_features_modules_0_parameters_weight_
    l_self_modules_layer_modules_features_modules_1_buffers_running_mean_ = L_self_modules_layer_modules_features_modules_1_buffers_running_mean_
    l_self_modules_layer_modules_features_modules_1_buffers_running_var_ = L_self_modules_layer_modules_features_modules_1_buffers_running_var_
    l_self_modules_layer_modules_features_modules_1_parameters_weight_ = L_self_modules_layer_modules_features_modules_1_parameters_weight_
    l_self_modules_layer_modules_features_modules_1_parameters_bias_ = L_self_modules_layer_modules_features_modules_1_parameters_bias_
    l_self_modules_layer_modules_features_modules_4_parameters_weight_ = L_self_modules_layer_modules_features_modules_4_parameters_weight_
    l_self_modules_layer_modules_features_modules_5_buffers_running_mean_ = L_self_modules_layer_modules_features_modules_5_buffers_running_mean_
    l_self_modules_layer_modules_features_modules_5_buffers_running_var_ = L_self_modules_layer_modules_features_modules_5_buffers_running_var_
    l_self_modules_layer_modules_features_modules_5_parameters_weight_ = L_self_modules_layer_modules_features_modules_5_parameters_weight_
    l_self_modules_layer_modules_features_modules_5_parameters_bias_ = L_self_modules_layer_modules_features_modules_5_parameters_bias_
    l_self_modules_layer_modules_features_modules_8_parameters_weight_ = L_self_modules_layer_modules_features_modules_8_parameters_weight_
    l_self_modules_layer_modules_features_modules_9_buffers_running_mean_ = L_self_modules_layer_modules_features_modules_9_buffers_running_mean_
    l_self_modules_layer_modules_features_modules_9_buffers_running_var_ = L_self_modules_layer_modules_features_modules_9_buffers_running_var_
    l_self_modules_layer_modules_features_modules_9_parameters_weight_ = L_self_modules_layer_modules_features_modules_9_parameters_weight_
    l_self_modules_layer_modules_features_modules_9_parameters_bias_ = L_self_modules_layer_modules_features_modules_9_parameters_bias_
    l_self_modules_layer_modules_features_modules_11_parameters_weight_ = L_self_modules_layer_modules_features_modules_11_parameters_weight_
    l_self_modules_layer_modules_features_modules_12_buffers_running_mean_ = L_self_modules_layer_modules_features_modules_12_buffers_running_mean_
    l_self_modules_layer_modules_features_modules_12_buffers_running_var_ = L_self_modules_layer_modules_features_modules_12_buffers_running_var_
    l_self_modules_layer_modules_features_modules_12_parameters_weight_ = L_self_modules_layer_modules_features_modules_12_parameters_weight_
    l_self_modules_layer_modules_features_modules_12_parameters_bias_ = L_self_modules_layer_modules_features_modules_12_parameters_bias_
    l_self_modules_layer_modules_features_modules_14_parameters_weight_ = L_self_modules_layer_modules_features_modules_14_parameters_weight_
    l_self_modules_layer_modules_features_modules_15_buffers_running_mean_ = L_self_modules_layer_modules_features_modules_15_buffers_running_mean_
    l_self_modules_layer_modules_features_modules_15_buffers_running_var_ = L_self_modules_layer_modules_features_modules_15_buffers_running_var_
    l_self_modules_layer_modules_features_modules_15_parameters_weight_ = L_self_modules_layer_modules_features_modules_15_parameters_weight_
    l_self_modules_layer_modules_features_modules_15_parameters_bias_ = L_self_modules_layer_modules_features_modules_15_parameters_bias_
    l_self_modules_layer_modules_classifier_modules_1_parameters_weight_ = L_self_modules_layer_modules_classifier_modules_1_parameters_weight_
    l_self_modules_layer_modules_classifier_modules_4_parameters_weight_ = L_self_modules_layer_modules_classifier_modules_4_parameters_weight_
    l_self_modules_layer_modules_classifier_modules_6_parameters_weight_ = L_self_modules_layer_modules_classifier_modules_6_parameters_weight_
    input_1_0_window_slice = torch.narrow(l_x_seq_, 0, 0, 16);  l_x_seq_ = None
    input_1_0_temporal_stack_flatten = torch.flatten(input_1_0_window_slice, 0, 1);  input_1_0_window_slice = None
    input_1_spatial_batch_conv = torch.conv2d(input_1_0_temporal_stack_flatten, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  input_1_0_temporal_stack_flatten = l_self_modules_layer_modules_features_modules_0_parameters_weight_ = None
    input_2_spatial_batch_bn = torch.nn.functional.batch_norm(input_1_spatial_batch_conv, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_1_spatial_batch_conv = l_self_modules_layer_modules_features_modules_1_buffers_running_mean_ = l_self_modules_layer_modules_features_modules_1_buffers_running_var_ = l_self_modules_layer_modules_features_modules_1_parameters_weight_ = l_self_modules_layer_modules_features_modules_1_parameters_bias_ = None
    input_2_spatial_batch_bn_chunks = torch.chunk(input_2_spatial_batch_bn, 16, 0)
    input_2_spatial_batch_bn_t0 = input_2_spatial_batch_bn_chunks[0];  input_2_spatial_batch_bn_chunks = None
    zeros_like = torch.zeros_like(input_2_spatial_batch_bn_t0);  input_2_spatial_batch_bn_t0 = None
    unflatten = input_2_spatial_batch_bn.unflatten(0, (16, -1));  input_2_spatial_batch_bn = None
    lif_forward_state_default_temporal_fused_lif_state = torch.ops.snn_custom.fused_temporal_lif_state.default(unflatten, zeros_like, 1.0, 0.0, 2.0, False);  unflatten = zeros_like = None
    lif_forward_state_default_temporal_fused_lif_state_spike_stack = lif_forward_state_default_temporal_fused_lif_state[0]
    lif_forward_state_default_temporal_fused_lif_state_v_final = lif_forward_state_default_temporal_fused_lif_state[1];  lif_forward_state_default_temporal_fused_lif_state = None
    input_3_0_temporal_stack_flatten = torch.flatten(lif_forward_state_default_temporal_fused_lif_state_spike_stack, 0, 1);  lif_forward_state_default_temporal_fused_lif_state_spike_stack = None
    input_3_spatial_batch_maxpool = torch.nn.functional.max_pool2d(input_3_0_temporal_stack_flatten, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  input_3_0_temporal_stack_flatten = None
    input_3_spatial_batch_maxpool_chunks = torch.chunk(input_3_spatial_batch_maxpool, 16, 0);  input_3_spatial_batch_maxpool = None
    input_3_spatial_batch_maxpool_t0 = input_3_spatial_batch_maxpool_chunks[0]
    input_3_spatial_batch_maxpool_t1 = input_3_spatial_batch_maxpool_chunks[1]
    input_3_spatial_batch_maxpool_t2 = input_3_spatial_batch_maxpool_chunks[2]
    input_3_spatial_batch_maxpool_t3 = input_3_spatial_batch_maxpool_chunks[3]
    input_3_spatial_batch_maxpool_t4 = input_3_spatial_batch_maxpool_chunks[4]
    input_3_spatial_batch_maxpool_t5 = input_3_spatial_batch_maxpool_chunks[5]
    input_3_spatial_batch_maxpool_t6 = input_3_spatial_batch_maxpool_chunks[6]
    input_3_spatial_batch_maxpool_t7 = input_3_spatial_batch_maxpool_chunks[7]
    input_3_spatial_batch_maxpool_t8 = input_3_spatial_batch_maxpool_chunks[8]
    input_3_spatial_batch_maxpool_t9 = input_3_spatial_batch_maxpool_chunks[9]
    input_3_spatial_batch_maxpool_t10 = input_3_spatial_batch_maxpool_chunks[10]
    input_3_spatial_batch_maxpool_t11 = input_3_spatial_batch_maxpool_chunks[11]
    input_3_spatial_batch_maxpool_t12 = input_3_spatial_batch_maxpool_chunks[12]
    input_3_spatial_batch_maxpool_t13 = input_3_spatial_batch_maxpool_chunks[13]
    input_3_spatial_batch_maxpool_t14 = input_3_spatial_batch_maxpool_chunks[14]
    input_3_spatial_batch_maxpool_t15 = input_3_spatial_batch_maxpool_chunks[15];  input_3_spatial_batch_maxpool_chunks = None
    _fx_zero_scalar_v_init_0 = self._fx_zero_scalar_v_init_0
    _fx_temporal_folded_conv_bn_weight_1 = self._fx_temporal_folded_conv_bn_weight_1
    _fx_temporal_folded_conv_bn_bias_2 = self._fx_temporal_folded_conv_bn_bias_2
    input_4_temporal_fused_regular_conv_lif_state = torch.ops.snn_custom.fused_temporal_conv_lif_state.default([input_3_spatial_batch_maxpool_t0, input_3_spatial_batch_maxpool_t1, input_3_spatial_batch_maxpool_t2, input_3_spatial_batch_maxpool_t3, input_3_spatial_batch_maxpool_t4, input_3_spatial_batch_maxpool_t5, input_3_spatial_batch_maxpool_t6, input_3_spatial_batch_maxpool_t7, input_3_spatial_batch_maxpool_t8, input_3_spatial_batch_maxpool_t9, input_3_spatial_batch_maxpool_t10, input_3_spatial_batch_maxpool_t11, input_3_spatial_batch_maxpool_t12, input_3_spatial_batch_maxpool_t13, input_3_spatial_batch_maxpool_t14, input_3_spatial_batch_maxpool_t15], _fx_temporal_folded_conv_bn_weight_1, _fx_temporal_folded_conv_bn_bias_2, _fx_zero_scalar_v_init_0, [1, 1], [2, 2], [1, 1], 1, 1.0, 0.0, 2.0, False);  input_3_spatial_batch_maxpool_t0 = input_3_spatial_batch_maxpool_t1 = input_3_spatial_batch_maxpool_t2 = input_3_spatial_batch_maxpool_t3 = input_3_spatial_batch_maxpool_t4 = input_3_spatial_batch_maxpool_t5 = input_3_spatial_batch_maxpool_t6 = input_3_spatial_batch_maxpool_t7 = input_3_spatial_batch_maxpool_t8 = input_3_spatial_batch_maxpool_t9 = input_3_spatial_batch_maxpool_t10 = input_3_spatial_batch_maxpool_t11 = input_3_spatial_batch_maxpool_t12 = input_3_spatial_batch_maxpool_t13 = input_3_spatial_batch_maxpool_t14 = input_3_spatial_batch_maxpool_t15 = _fx_temporal_folded_conv_bn_weight_1 = _fx_temporal_folded_conv_bn_bias_2 = _fx_zero_scalar_v_init_0 = None
    input_4_temporal_fused_regular_conv_lif_state_spike_stack = input_4_temporal_fused_regular_conv_lif_state[0]
    input_4_temporal_fused_regular_conv_lif_state_v_final = input_4_temporal_fused_regular_conv_lif_state[1];  input_4_temporal_fused_regular_conv_lif_state = None
    input_6_0_temporal_stack_flatten = torch.flatten(input_4_temporal_fused_regular_conv_lif_state_spike_stack, 0, 1);  input_4_temporal_fused_regular_conv_lif_state_spike_stack = None
    input_6_spatial_batch_maxpool = torch.nn.functional.max_pool2d(input_6_0_temporal_stack_flatten, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  input_6_0_temporal_stack_flatten = None
    input_6_spatial_batch_maxpool_chunks = torch.chunk(input_6_spatial_batch_maxpool, 16, 0);  input_6_spatial_batch_maxpool = None
    input_6_spatial_batch_maxpool_t0 = input_6_spatial_batch_maxpool_chunks[0]
    input_6_spatial_batch_maxpool_t1 = input_6_spatial_batch_maxpool_chunks[1]
    input_6_spatial_batch_maxpool_t2 = input_6_spatial_batch_maxpool_chunks[2]
    input_6_spatial_batch_maxpool_t3 = input_6_spatial_batch_maxpool_chunks[3]
    input_6_spatial_batch_maxpool_t4 = input_6_spatial_batch_maxpool_chunks[4]
    input_6_spatial_batch_maxpool_t5 = input_6_spatial_batch_maxpool_chunks[5]
    input_6_spatial_batch_maxpool_t6 = input_6_spatial_batch_maxpool_chunks[6]
    input_6_spatial_batch_maxpool_t7 = input_6_spatial_batch_maxpool_chunks[7]
    input_6_spatial_batch_maxpool_t8 = input_6_spatial_batch_maxpool_chunks[8]
    input_6_spatial_batch_maxpool_t9 = input_6_spatial_batch_maxpool_chunks[9]
    input_6_spatial_batch_maxpool_t10 = input_6_spatial_batch_maxpool_chunks[10]
    input_6_spatial_batch_maxpool_t11 = input_6_spatial_batch_maxpool_chunks[11]
    input_6_spatial_batch_maxpool_t12 = input_6_spatial_batch_maxpool_chunks[12]
    input_6_spatial_batch_maxpool_t13 = input_6_spatial_batch_maxpool_chunks[13]
    input_6_spatial_batch_maxpool_t14 = input_6_spatial_batch_maxpool_chunks[14]
    input_6_spatial_batch_maxpool_t15 = input_6_spatial_batch_maxpool_chunks[15];  input_6_spatial_batch_maxpool_chunks = None
    _fx_zero_scalar_v_init_3 = self._fx_zero_scalar_v_init_3
    _fx_temporal_folded_conv_bn_weight_4 = self._fx_temporal_folded_conv_bn_weight_4
    _fx_temporal_folded_conv_bn_bias_5 = self._fx_temporal_folded_conv_bn_bias_5
    input_7_temporal_fused_regular_conv_lif_state = torch.ops.snn_custom.fused_temporal_conv_lif_state.default([input_6_spatial_batch_maxpool_t0, input_6_spatial_batch_maxpool_t1, input_6_spatial_batch_maxpool_t2, input_6_spatial_batch_maxpool_t3, input_6_spatial_batch_maxpool_t4, input_6_spatial_batch_maxpool_t5, input_6_spatial_batch_maxpool_t6, input_6_spatial_batch_maxpool_t7, input_6_spatial_batch_maxpool_t8, input_6_spatial_batch_maxpool_t9, input_6_spatial_batch_maxpool_t10, input_6_spatial_batch_maxpool_t11, input_6_spatial_batch_maxpool_t12, input_6_spatial_batch_maxpool_t13, input_6_spatial_batch_maxpool_t14, input_6_spatial_batch_maxpool_t15], _fx_temporal_folded_conv_bn_weight_4, _fx_temporal_folded_conv_bn_bias_5, _fx_zero_scalar_v_init_3, [1, 1], [1, 1], [1, 1], 1, 1.0, 0.0, 2.0, False);  input_6_spatial_batch_maxpool_t0 = input_6_spatial_batch_maxpool_t1 = input_6_spatial_batch_maxpool_t2 = input_6_spatial_batch_maxpool_t3 = input_6_spatial_batch_maxpool_t4 = input_6_spatial_batch_maxpool_t5 = input_6_spatial_batch_maxpool_t6 = input_6_spatial_batch_maxpool_t7 = input_6_spatial_batch_maxpool_t8 = input_6_spatial_batch_maxpool_t9 = input_6_spatial_batch_maxpool_t10 = input_6_spatial_batch_maxpool_t11 = input_6_spatial_batch_maxpool_t12 = input_6_spatial_batch_maxpool_t13 = input_6_spatial_batch_maxpool_t14 = input_6_spatial_batch_maxpool_t15 = _fx_temporal_folded_conv_bn_weight_4 = _fx_temporal_folded_conv_bn_bias_5 = _fx_zero_scalar_v_init_3 = None
    input_7_temporal_fused_regular_conv_lif_state_spike_stack = input_7_temporal_fused_regular_conv_lif_state[0]
    input_7_temporal_fused_regular_conv_lif_state_v_final = input_7_temporal_fused_regular_conv_lif_state[1];  input_7_temporal_fused_regular_conv_lif_state = None
    input_7_temporal_fused_regular_conv_lif_state_spike_t0 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[0]
    input_7_temporal_fused_regular_conv_lif_state_spike_t1 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[1]
    input_7_temporal_fused_regular_conv_lif_state_spike_t2 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[2]
    input_7_temporal_fused_regular_conv_lif_state_spike_t3 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[3]
    input_7_temporal_fused_regular_conv_lif_state_spike_t4 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[4]
    input_7_temporal_fused_regular_conv_lif_state_spike_t5 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[5]
    input_7_temporal_fused_regular_conv_lif_state_spike_t6 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[6]
    input_7_temporal_fused_regular_conv_lif_state_spike_t7 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[7]
    input_7_temporal_fused_regular_conv_lif_state_spike_t8 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[8]
    input_7_temporal_fused_regular_conv_lif_state_spike_t9 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[9]
    input_7_temporal_fused_regular_conv_lif_state_spike_t10 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[10]
    input_7_temporal_fused_regular_conv_lif_state_spike_t11 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[11]
    input_7_temporal_fused_regular_conv_lif_state_spike_t12 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[12]
    input_7_temporal_fused_regular_conv_lif_state_spike_t13 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[13]
    input_7_temporal_fused_regular_conv_lif_state_spike_t14 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[14]
    input_7_temporal_fused_regular_conv_lif_state_spike_t15 = input_7_temporal_fused_regular_conv_lif_state_spike_stack[15];  input_7_temporal_fused_regular_conv_lif_state_spike_stack = None
    _fx_zero_scalar_v_init_6 = self._fx_zero_scalar_v_init_6
    _fx_temporal_folded_conv_bn_weight_7 = self._fx_temporal_folded_conv_bn_weight_7
    _fx_temporal_folded_conv_bn_bias_8 = self._fx_temporal_folded_conv_bn_bias_8
    input_9_temporal_fused_regular_conv_lif_state = torch.ops.snn_custom.fused_temporal_conv_lif_state.default([input_7_temporal_fused_regular_conv_lif_state_spike_t0, input_7_temporal_fused_regular_conv_lif_state_spike_t1, input_7_temporal_fused_regular_conv_lif_state_spike_t2, input_7_temporal_fused_regular_conv_lif_state_spike_t3, input_7_temporal_fused_regular_conv_lif_state_spike_t4, input_7_temporal_fused_regular_conv_lif_state_spike_t5, input_7_temporal_fused_regular_conv_lif_state_spike_t6, input_7_temporal_fused_regular_conv_lif_state_spike_t7, input_7_temporal_fused_regular_conv_lif_state_spike_t8, input_7_temporal_fused_regular_conv_lif_state_spike_t9, input_7_temporal_fused_regular_conv_lif_state_spike_t10, input_7_temporal_fused_regular_conv_lif_state_spike_t11, input_7_temporal_fused_regular_conv_lif_state_spike_t12, input_7_temporal_fused_regular_conv_lif_state_spike_t13, input_7_temporal_fused_regular_conv_lif_state_spike_t14, input_7_temporal_fused_regular_conv_lif_state_spike_t15], _fx_temporal_folded_conv_bn_weight_7, _fx_temporal_folded_conv_bn_bias_8, _fx_zero_scalar_v_init_6, [1, 1], [1, 1], [1, 1], 1, 1.0, 0.0, 2.0, False);  input_7_temporal_fused_regular_conv_lif_state_spike_t0 = input_7_temporal_fused_regular_conv_lif_state_spike_t1 = input_7_temporal_fused_regular_conv_lif_state_spike_t2 = input_7_temporal_fused_regular_conv_lif_state_spike_t3 = input_7_temporal_fused_regular_conv_lif_state_spike_t4 = input_7_temporal_fused_regular_conv_lif_state_spike_t5 = input_7_temporal_fused_regular_conv_lif_state_spike_t6 = input_7_temporal_fused_regular_conv_lif_state_spike_t7 = input_7_temporal_fused_regular_conv_lif_state_spike_t8 = input_7_temporal_fused_regular_conv_lif_state_spike_t9 = input_7_temporal_fused_regular_conv_lif_state_spike_t10 = input_7_temporal_fused_regular_conv_lif_state_spike_t11 = input_7_temporal_fused_regular_conv_lif_state_spike_t12 = input_7_temporal_fused_regular_conv_lif_state_spike_t13 = input_7_temporal_fused_regular_conv_lif_state_spike_t14 = input_7_temporal_fused_regular_conv_lif_state_spike_t15 = _fx_temporal_folded_conv_bn_weight_7 = _fx_temporal_folded_conv_bn_bias_8 = _fx_zero_scalar_v_init_6 = None
    input_9_temporal_fused_regular_conv_lif_state_spike_stack = input_9_temporal_fused_regular_conv_lif_state[0]
    input_9_temporal_fused_regular_conv_lif_state_v_final = input_9_temporal_fused_regular_conv_lif_state[1];  input_9_temporal_fused_regular_conv_lif_state = None
    input_9_temporal_fused_regular_conv_lif_state_spike_t0 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[0]
    input_9_temporal_fused_regular_conv_lif_state_spike_t1 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[1]
    input_9_temporal_fused_regular_conv_lif_state_spike_t2 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[2]
    input_9_temporal_fused_regular_conv_lif_state_spike_t3 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[3]
    input_9_temporal_fused_regular_conv_lif_state_spike_t4 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[4]
    input_9_temporal_fused_regular_conv_lif_state_spike_t5 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[5]
    input_9_temporal_fused_regular_conv_lif_state_spike_t6 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[6]
    input_9_temporal_fused_regular_conv_lif_state_spike_t7 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[7]
    input_9_temporal_fused_regular_conv_lif_state_spike_t8 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[8]
    input_9_temporal_fused_regular_conv_lif_state_spike_t9 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[9]
    input_9_temporal_fused_regular_conv_lif_state_spike_t10 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[10]
    input_9_temporal_fused_regular_conv_lif_state_spike_t11 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[11]
    input_9_temporal_fused_regular_conv_lif_state_spike_t12 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[12]
    input_9_temporal_fused_regular_conv_lif_state_spike_t13 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[13]
    input_9_temporal_fused_regular_conv_lif_state_spike_t14 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[14]
    input_9_temporal_fused_regular_conv_lif_state_spike_t15 = input_9_temporal_fused_regular_conv_lif_state_spike_stack[15];  input_9_temporal_fused_regular_conv_lif_state_spike_stack = None
    _fx_zero_scalar_v_init_9 = self._fx_zero_scalar_v_init_9
    _fx_temporal_folded_conv_bn_weight_10 = self._fx_temporal_folded_conv_bn_weight_10
    _fx_temporal_folded_conv_bn_bias_11 = self._fx_temporal_folded_conv_bn_bias_11
    input_11_temporal_fused_regular_conv_lif_state = torch.ops.snn_custom.fused_temporal_conv_lif_state.default([input_9_temporal_fused_regular_conv_lif_state_spike_t0, input_9_temporal_fused_regular_conv_lif_state_spike_t1, input_9_temporal_fused_regular_conv_lif_state_spike_t2, input_9_temporal_fused_regular_conv_lif_state_spike_t3, input_9_temporal_fused_regular_conv_lif_state_spike_t4, input_9_temporal_fused_regular_conv_lif_state_spike_t5, input_9_temporal_fused_regular_conv_lif_state_spike_t6, input_9_temporal_fused_regular_conv_lif_state_spike_t7, input_9_temporal_fused_regular_conv_lif_state_spike_t8, input_9_temporal_fused_regular_conv_lif_state_spike_t9, input_9_temporal_fused_regular_conv_lif_state_spike_t10, input_9_temporal_fused_regular_conv_lif_state_spike_t11, input_9_temporal_fused_regular_conv_lif_state_spike_t12, input_9_temporal_fused_regular_conv_lif_state_spike_t13, input_9_temporal_fused_regular_conv_lif_state_spike_t14, input_9_temporal_fused_regular_conv_lif_state_spike_t15], _fx_temporal_folded_conv_bn_weight_10, _fx_temporal_folded_conv_bn_bias_11, _fx_zero_scalar_v_init_9, [1, 1], [1, 1], [1, 1], 1, 1.0, 0.0, 2.0, False);  input_9_temporal_fused_regular_conv_lif_state_spike_t0 = input_9_temporal_fused_regular_conv_lif_state_spike_t1 = input_9_temporal_fused_regular_conv_lif_state_spike_t2 = input_9_temporal_fused_regular_conv_lif_state_spike_t3 = input_9_temporal_fused_regular_conv_lif_state_spike_t4 = input_9_temporal_fused_regular_conv_lif_state_spike_t5 = input_9_temporal_fused_regular_conv_lif_state_spike_t6 = input_9_temporal_fused_regular_conv_lif_state_spike_t7 = input_9_temporal_fused_regular_conv_lif_state_spike_t8 = input_9_temporal_fused_regular_conv_lif_state_spike_t9 = input_9_temporal_fused_regular_conv_lif_state_spike_t10 = input_9_temporal_fused_regular_conv_lif_state_spike_t11 = input_9_temporal_fused_regular_conv_lif_state_spike_t12 = input_9_temporal_fused_regular_conv_lif_state_spike_t13 = input_9_temporal_fused_regular_conv_lif_state_spike_t14 = input_9_temporal_fused_regular_conv_lif_state_spike_t15 = _fx_temporal_folded_conv_bn_weight_10 = _fx_temporal_folded_conv_bn_bias_11 = _fx_zero_scalar_v_init_9 = None
    input_11_temporal_fused_regular_conv_lif_state_spike_stack = input_11_temporal_fused_regular_conv_lif_state[0]
    input_11_temporal_fused_regular_conv_lif_state_v_final = input_11_temporal_fused_regular_conv_lif_state[1];  input_11_temporal_fused_regular_conv_lif_state = None
    input_13_0_temporal_stack_flatten = torch.flatten(input_11_temporal_fused_regular_conv_lif_state_spike_stack, 0, 1);  input_11_temporal_fused_regular_conv_lif_state_spike_stack = None
    input_13_spatial_batch_maxpool = torch.nn.functional.max_pool2d(input_13_0_temporal_stack_flatten, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  input_13_0_temporal_stack_flatten = None
    x_spatial_batch_adaptive_avg_pool = torch.nn.functional.adaptive_avg_pool2d(input_13_spatial_batch_maxpool, (6, 6));  input_13_spatial_batch_maxpool = None
    x_1_spatial_batch_flatten = x_spatial_batch_adaptive_avg_pool.flatten(1, -1);  x_spatial_batch_adaptive_avg_pool = None
    x_1_spatial_batch_flatten_chunks = torch.chunk(x_1_spatial_batch_flatten, 16, 0);  x_1_spatial_batch_flatten = None
    x_1_spatial_batch_flatten_t0 = x_1_spatial_batch_flatten_chunks[0]
    x_1_spatial_batch_flatten_t1 = x_1_spatial_batch_flatten_chunks[1]
    x_1_spatial_batch_flatten_t2 = x_1_spatial_batch_flatten_chunks[2]
    x_1_spatial_batch_flatten_t3 = x_1_spatial_batch_flatten_chunks[3]
    x_1_spatial_batch_flatten_t4 = x_1_spatial_batch_flatten_chunks[4]
    x_1_spatial_batch_flatten_t5 = x_1_spatial_batch_flatten_chunks[5]
    x_1_spatial_batch_flatten_t6 = x_1_spatial_batch_flatten_chunks[6]
    x_1_spatial_batch_flatten_t7 = x_1_spatial_batch_flatten_chunks[7]
    x_1_spatial_batch_flatten_t8 = x_1_spatial_batch_flatten_chunks[8]
    x_1_spatial_batch_flatten_t9 = x_1_spatial_batch_flatten_chunks[9]
    x_1_spatial_batch_flatten_t10 = x_1_spatial_batch_flatten_chunks[10]
    x_1_spatial_batch_flatten_t11 = x_1_spatial_batch_flatten_chunks[11]
    x_1_spatial_batch_flatten_t12 = x_1_spatial_batch_flatten_chunks[12]
    x_1_spatial_batch_flatten_t13 = x_1_spatial_batch_flatten_chunks[13]
    x_1_spatial_batch_flatten_t14 = x_1_spatial_batch_flatten_chunks[14]
    x_1_spatial_batch_flatten_t15 = x_1_spatial_batch_flatten_chunks[15];  x_1_spatial_batch_flatten_chunks = None
    input_14 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t0, 0.5, False, False);  x_1_spatial_batch_flatten_t0 = None
    input_32 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t1, 0.5, False, False);  x_1_spatial_batch_flatten_t1 = None
    input_50 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t2, 0.5, False, False);  x_1_spatial_batch_flatten_t2 = None
    input_68 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t3, 0.5, False, False);  x_1_spatial_batch_flatten_t3 = None
    input_86 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t4, 0.5, False, False);  x_1_spatial_batch_flatten_t4 = None
    input_104 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t5, 0.5, False, False);  x_1_spatial_batch_flatten_t5 = None
    input_122 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t6, 0.5, False, False);  x_1_spatial_batch_flatten_t6 = None
    input_140 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t7, 0.5, False, False);  x_1_spatial_batch_flatten_t7 = None
    input_158 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t8, 0.5, False, False);  x_1_spatial_batch_flatten_t8 = None
    input_176 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t9, 0.5, False, False);  x_1_spatial_batch_flatten_t9 = None
    input_194 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t10, 0.5, False, False);  x_1_spatial_batch_flatten_t10 = None
    input_212 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t11, 0.5, False, False);  x_1_spatial_batch_flatten_t11 = None
    input_230 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t12, 0.5, False, False);  x_1_spatial_batch_flatten_t12 = None
    input_248 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t13, 0.5, False, False);  x_1_spatial_batch_flatten_t13 = None
    input_266 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t14, 0.5, False, False);  x_1_spatial_batch_flatten_t14 = None
    input_284 = torch.nn.functional.dropout(x_1_spatial_batch_flatten_t15, 0.5, False, False);  x_1_spatial_batch_flatten_t15 = None
    input_15_temporal_linear_lif_x_seq = torch.stack([input_14, input_32, input_50, input_68, input_86, input_104, input_122, input_140, input_158, input_176, input_194, input_212, input_230, input_248, input_266, input_284], 0);  input_14 = input_32 = input_50 = input_68 = input_86 = input_104 = input_122 = input_140 = input_158 = input_176 = input_194 = input_212 = input_230 = input_248 = input_266 = input_284 = None
    input_15_temporal_linear_lif_x_seq_shape = input_15_temporal_linear_lif_x_seq.size()
    size_1 = l_self_modules_layer_modules_classifier_modules_1_parameters_weight_.size(0)
    input_15_temporal_linear_lif_x_seq_spike_prefix = input_15_temporal_linear_lif_x_seq_shape[slice(None, -1, None)]
    input_15_temporal_linear_lif_x_seq_spike_shape = input_15_temporal_linear_lif_x_seq_spike_prefix + (size_1,);  input_15_temporal_linear_lif_x_seq_spike_prefix = None
    input_15_temporal_linear_lif_x_seq_v_prefix = input_15_temporal_linear_lif_x_seq_shape[slice(1, -1, None)];  input_15_temporal_linear_lif_x_seq_shape = None
    input_15_temporal_linear_lif_x_seq_v_shape = input_15_temporal_linear_lif_x_seq_v_prefix + (size_1,);  input_15_temporal_linear_lif_x_seq_v_prefix = size_1 = None
    input_15_temporal_fused_linear_lif_state_spike_stack = input_15_temporal_linear_lif_x_seq.new_empty(input_15_temporal_linear_lif_x_seq_spike_shape);  input_15_temporal_linear_lif_x_seq_spike_shape = None
    input_15_temporal_fused_linear_lif_state_v_final = input_15_temporal_linear_lif_x_seq.new_empty(input_15_temporal_linear_lif_x_seq_v_shape);  input_15_temporal_linear_lif_x_seq_v_shape = None
    input_15_temporal_linear_lif_v_init_device = input_15_temporal_linear_lif_x_seq.new_zeros(())
    input_15_temporal_fused_linear_lif_state_out = torch.ops.snn_custom.fused_temporal_linear_lif_state_packed_out.default(input_15_temporal_linear_lif_x_seq, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None, input_15_temporal_linear_lif_v_init_device, 1.0, 0.0, 2.0, False, input_15_temporal_fused_linear_lif_state_spike_stack, input_15_temporal_fused_linear_lif_state_v_final);  input_15_temporal_linear_lif_x_seq = l_self_modules_layer_modules_classifier_modules_1_parameters_weight_ = input_15_temporal_linear_lif_v_init_device = input_15_temporal_fused_linear_lif_state_out = None
    input_15_temporal_fused_linear_lif_state_v_final_pool_safe = torch.clone(input_15_temporal_fused_linear_lif_state_v_final);  input_15_temporal_fused_linear_lif_state_v_final = None
    input_15_temporal_fused_linear_lif_state_out_spike_t0 = input_15_temporal_fused_linear_lif_state_spike_stack[0]
    input_15_temporal_fused_linear_lif_state_out_spike_t1 = input_15_temporal_fused_linear_lif_state_spike_stack[1]
    input_15_temporal_fused_linear_lif_state_out_spike_t2 = input_15_temporal_fused_linear_lif_state_spike_stack[2]
    input_15_temporal_fused_linear_lif_state_out_spike_t3 = input_15_temporal_fused_linear_lif_state_spike_stack[3]
    input_15_temporal_fused_linear_lif_state_out_spike_t4 = input_15_temporal_fused_linear_lif_state_spike_stack[4]
    input_15_temporal_fused_linear_lif_state_out_spike_t5 = input_15_temporal_fused_linear_lif_state_spike_stack[5]
    input_15_temporal_fused_linear_lif_state_out_spike_t6 = input_15_temporal_fused_linear_lif_state_spike_stack[6]
    input_15_temporal_fused_linear_lif_state_out_spike_t7 = input_15_temporal_fused_linear_lif_state_spike_stack[7]
    input_15_temporal_fused_linear_lif_state_out_spike_t8 = input_15_temporal_fused_linear_lif_state_spike_stack[8]
    input_15_temporal_fused_linear_lif_state_out_spike_t9 = input_15_temporal_fused_linear_lif_state_spike_stack[9]
    input_15_temporal_fused_linear_lif_state_out_spike_t10 = input_15_temporal_fused_linear_lif_state_spike_stack[10]
    input_15_temporal_fused_linear_lif_state_out_spike_t11 = input_15_temporal_fused_linear_lif_state_spike_stack[11]
    input_15_temporal_fused_linear_lif_state_out_spike_t12 = input_15_temporal_fused_linear_lif_state_spike_stack[12]
    input_15_temporal_fused_linear_lif_state_out_spike_t13 = input_15_temporal_fused_linear_lif_state_spike_stack[13]
    input_15_temporal_fused_linear_lif_state_out_spike_t14 = input_15_temporal_fused_linear_lif_state_spike_stack[14]
    input_15_temporal_fused_linear_lif_state_out_spike_t15 = input_15_temporal_fused_linear_lif_state_spike_stack[15];  input_15_temporal_fused_linear_lif_state_spike_stack = None
    input_16 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t0, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t0 = None
    input_34 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t1, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t1 = None
    input_52 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t2, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t2 = None
    input_70 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t3, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t3 = None
    input_88 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t4, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t4 = None
    input_106 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t5, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t5 = None
    input_124 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t6, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t6 = None
    input_142 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t7, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t7 = None
    input_160 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t8, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t8 = None
    input_178 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t9, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t9 = None
    input_196 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t10, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t10 = None
    input_214 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t11, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t11 = None
    input_232 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t12, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t12 = None
    input_250 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t13, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t13 = None
    input_268 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t14, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t14 = None
    input_286 = torch.nn.functional.dropout(input_15_temporal_fused_linear_lif_state_out_spike_t15, 0.5, False, False);  input_15_temporal_fused_linear_lif_state_out_spike_t15 = None
    input_17_temporal_linear_lif_x_seq = torch.stack([input_16, input_34, input_52, input_70, input_88, input_106, input_124, input_142, input_160, input_178, input_196, input_214, input_232, input_250, input_268, input_286], 0);  input_16 = input_34 = input_52 = input_70 = input_88 = input_106 = input_124 = input_142 = input_160 = input_178 = input_196 = input_214 = input_232 = input_250 = input_268 = input_286 = None
    input_17_temporal_linear_lif_x_seq_shape = input_17_temporal_linear_lif_x_seq.size()
    size_3 = l_self_modules_layer_modules_classifier_modules_4_parameters_weight_.size(0)
    input_17_temporal_linear_lif_x_seq_spike_prefix = input_17_temporal_linear_lif_x_seq_shape[slice(None, -1, None)]
    input_17_temporal_linear_lif_x_seq_spike_shape = input_17_temporal_linear_lif_x_seq_spike_prefix + (size_3,);  input_17_temporal_linear_lif_x_seq_spike_prefix = None
    input_17_temporal_linear_lif_x_seq_v_prefix = input_17_temporal_linear_lif_x_seq_shape[slice(1, -1, None)];  input_17_temporal_linear_lif_x_seq_shape = None
    input_17_temporal_linear_lif_x_seq_v_shape = input_17_temporal_linear_lif_x_seq_v_prefix + (size_3,);  input_17_temporal_linear_lif_x_seq_v_prefix = size_3 = None
    input_17_temporal_fused_linear_lif_state_spike_stack = input_17_temporal_linear_lif_x_seq.new_empty(input_17_temporal_linear_lif_x_seq_spike_shape);  input_17_temporal_linear_lif_x_seq_spike_shape = None
    input_17_temporal_fused_linear_lif_state_v_final = input_17_temporal_linear_lif_x_seq.new_empty(input_17_temporal_linear_lif_x_seq_v_shape);  input_17_temporal_linear_lif_x_seq_v_shape = None
    input_17_temporal_linear_lif_v_init_device = input_17_temporal_linear_lif_x_seq.new_zeros(())
    input_17_temporal_fused_linear_lif_state_out = torch.ops.snn_custom.fused_temporal_linear_lif_state_packed_out.default(input_17_temporal_linear_lif_x_seq, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None, input_17_temporal_linear_lif_v_init_device, 1.0, 0.0, 2.0, False, input_17_temporal_fused_linear_lif_state_spike_stack, input_17_temporal_fused_linear_lif_state_v_final);  input_17_temporal_linear_lif_x_seq = l_self_modules_layer_modules_classifier_modules_4_parameters_weight_ = input_17_temporal_linear_lif_v_init_device = input_17_temporal_fused_linear_lif_state_out = None
    input_17_temporal_fused_linear_lif_state_v_final_pool_safe = torch.clone(input_17_temporal_fused_linear_lif_state_v_final);  input_17_temporal_fused_linear_lif_state_v_final = None
    input_18_temporal_linear_lif_x_seq_shape = input_17_temporal_fused_linear_lif_state_spike_stack.size()
    size_5 = l_self_modules_layer_modules_classifier_modules_6_parameters_weight_.size(0)
    input_18_temporal_linear_lif_x_seq_spike_prefix = input_18_temporal_linear_lif_x_seq_shape[slice(None, -1, None)]
    input_18_temporal_linear_lif_x_seq_spike_shape = input_18_temporal_linear_lif_x_seq_spike_prefix + (size_5,);  input_18_temporal_linear_lif_x_seq_spike_prefix = None
    input_18_temporal_linear_lif_x_seq_v_prefix = input_18_temporal_linear_lif_x_seq_shape[slice(1, -1, None)];  input_18_temporal_linear_lif_x_seq_shape = None
    input_18_temporal_linear_lif_x_seq_v_shape = input_18_temporal_linear_lif_x_seq_v_prefix + (size_5,);  input_18_temporal_linear_lif_x_seq_v_prefix = size_5 = None
    input_18_temporal_fused_linear_lif_state_spike_stack = input_17_temporal_fused_linear_lif_state_spike_stack.new_empty(input_18_temporal_linear_lif_x_seq_spike_shape);  input_18_temporal_linear_lif_x_seq_spike_shape = None
    input_18_temporal_fused_linear_lif_state_v_final = input_17_temporal_fused_linear_lif_state_spike_stack.new_empty(input_18_temporal_linear_lif_x_seq_v_shape);  input_18_temporal_linear_lif_x_seq_v_shape = None
    input_18_temporal_linear_lif_v_init_device = input_17_temporal_fused_linear_lif_state_spike_stack.new_zeros(())
    input_18_temporal_fused_linear_lif_state_out = torch.ops.snn_custom.fused_temporal_linear_lif_state_packed_out.default(input_17_temporal_fused_linear_lif_state_spike_stack, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None, input_18_temporal_linear_lif_v_init_device, 1.0, 0.0, 2.0, False, input_18_temporal_fused_linear_lif_state_spike_stack, input_18_temporal_fused_linear_lif_state_v_final);  input_17_temporal_fused_linear_lif_state_spike_stack = l_self_modules_layer_modules_classifier_modules_6_parameters_weight_ = input_18_temporal_linear_lif_v_init_device = input_18_temporal_fused_linear_lif_state_out = None
    input_18_temporal_fused_linear_lif_state_v_final_pool_safe = torch.clone(input_18_temporal_fused_linear_lif_state_v_final);  input_18_temporal_fused_linear_lif_state_v_final = None
    input_18_temporal_fused_linear_lif_state_out_spike_t0 = input_18_temporal_fused_linear_lif_state_spike_stack[0]
    input_18_temporal_fused_linear_lif_state_out_spike_t1 = input_18_temporal_fused_linear_lif_state_spike_stack[1]
    input_18_temporal_fused_linear_lif_state_out_spike_t2 = input_18_temporal_fused_linear_lif_state_spike_stack[2]
    input_18_temporal_fused_linear_lif_state_out_spike_t3 = input_18_temporal_fused_linear_lif_state_spike_stack[3]
    input_18_temporal_fused_linear_lif_state_out_spike_t4 = input_18_temporal_fused_linear_lif_state_spike_stack[4]
    input_18_temporal_fused_linear_lif_state_out_spike_t5 = input_18_temporal_fused_linear_lif_state_spike_stack[5]
    input_18_temporal_fused_linear_lif_state_out_spike_t6 = input_18_temporal_fused_linear_lif_state_spike_stack[6]
    input_18_temporal_fused_linear_lif_state_out_spike_t7 = input_18_temporal_fused_linear_lif_state_spike_stack[7]
    input_18_temporal_fused_linear_lif_state_out_spike_t8 = input_18_temporal_fused_linear_lif_state_spike_stack[8]
    input_18_temporal_fused_linear_lif_state_out_spike_t9 = input_18_temporal_fused_linear_lif_state_spike_stack[9]
    input_18_temporal_fused_linear_lif_state_out_spike_t10 = input_18_temporal_fused_linear_lif_state_spike_stack[10]
    input_18_temporal_fused_linear_lif_state_out_spike_t11 = input_18_temporal_fused_linear_lif_state_spike_stack[11]
    input_18_temporal_fused_linear_lif_state_out_spike_t12 = input_18_temporal_fused_linear_lif_state_spike_stack[12]
    input_18_temporal_fused_linear_lif_state_out_spike_t13 = input_18_temporal_fused_linear_lif_state_spike_stack[13]
    input_18_temporal_fused_linear_lif_state_out_spike_t14 = input_18_temporal_fused_linear_lif_state_spike_stack[14]
    input_18_temporal_fused_linear_lif_state_out_spike_t15 = input_18_temporal_fused_linear_lif_state_spike_stack[15];  input_18_temporal_fused_linear_lif_state_spike_stack = None
    out_spikes_counter = 0 + input_18_temporal_fused_linear_lif_state_out_spike_t0;  input_18_temporal_fused_linear_lif_state_out_spike_t0 = None
    out_spikes_counter_1 = out_spikes_counter + input_18_temporal_fused_linear_lif_state_out_spike_t1;  out_spikes_counter = input_18_temporal_fused_linear_lif_state_out_spike_t1 = None
    out_spikes_counter_2 = out_spikes_counter_1 + input_18_temporal_fused_linear_lif_state_out_spike_t2;  out_spikes_counter_1 = input_18_temporal_fused_linear_lif_state_out_spike_t2 = None
    out_spikes_counter_3 = out_spikes_counter_2 + input_18_temporal_fused_linear_lif_state_out_spike_t3;  out_spikes_counter_2 = input_18_temporal_fused_linear_lif_state_out_spike_t3 = None
    out_spikes_counter_4 = out_spikes_counter_3 + input_18_temporal_fused_linear_lif_state_out_spike_t4;  out_spikes_counter_3 = input_18_temporal_fused_linear_lif_state_out_spike_t4 = None
    out_spikes_counter_5 = out_spikes_counter_4 + input_18_temporal_fused_linear_lif_state_out_spike_t5;  out_spikes_counter_4 = input_18_temporal_fused_linear_lif_state_out_spike_t5 = None
    out_spikes_counter_6 = out_spikes_counter_5 + input_18_temporal_fused_linear_lif_state_out_spike_t6;  out_spikes_counter_5 = input_18_temporal_fused_linear_lif_state_out_spike_t6 = None
    out_spikes_counter_7 = out_spikes_counter_6 + input_18_temporal_fused_linear_lif_state_out_spike_t7;  out_spikes_counter_6 = input_18_temporal_fused_linear_lif_state_out_spike_t7 = None
    out_spikes_counter_8 = out_spikes_counter_7 + input_18_temporal_fused_linear_lif_state_out_spike_t8;  out_spikes_counter_7 = input_18_temporal_fused_linear_lif_state_out_spike_t8 = None
    out_spikes_counter_9 = out_spikes_counter_8 + input_18_temporal_fused_linear_lif_state_out_spike_t9;  out_spikes_counter_8 = input_18_temporal_fused_linear_lif_state_out_spike_t9 = None
    out_spikes_counter_10 = out_spikes_counter_9 + input_18_temporal_fused_linear_lif_state_out_spike_t10;  out_spikes_counter_9 = input_18_temporal_fused_linear_lif_state_out_spike_t10 = None
    out_spikes_counter_11 = out_spikes_counter_10 + input_18_temporal_fused_linear_lif_state_out_spike_t11;  out_spikes_counter_10 = input_18_temporal_fused_linear_lif_state_out_spike_t11 = None
    out_spikes_counter_12 = out_spikes_counter_11 + input_18_temporal_fused_linear_lif_state_out_spike_t12;  out_spikes_counter_11 = input_18_temporal_fused_linear_lif_state_out_spike_t12 = None
    out_spikes_counter_13 = out_spikes_counter_12 + input_18_temporal_fused_linear_lif_state_out_spike_t13;  out_spikes_counter_12 = input_18_temporal_fused_linear_lif_state_out_spike_t13 = None
    out_spikes_counter_14 = out_spikes_counter_13 + input_18_temporal_fused_linear_lif_state_out_spike_t14;  out_spikes_counter_13 = input_18_temporal_fused_linear_lif_state_out_spike_t14 = None
    out_spikes_counter_15 = out_spikes_counter_14 + input_18_temporal_fused_linear_lif_state_out_spike_t15;  out_spikes_counter_14 = input_18_temporal_fused_linear_lif_state_out_spike_t15 = None
    truediv = out_spikes_counter_15 / 16;  out_spikes_counter_15 = None
    return (truediv, lif_forward_state_default_temporal_fused_lif_state_v_final, input_4_temporal_fused_regular_conv_lif_state_v_final, input_7_temporal_fused_regular_conv_lif_state_v_final, input_9_temporal_fused_regular_conv_lif_state_v_final, input_11_temporal_fused_regular_conv_lif_state_v_final, input_15_temporal_fused_linear_lif_state_v_final_pool_safe, input_17_temporal_fused_linear_lif_state_v_final_pool_safe, input_18_temporal_fused_linear_lif_state_v_final_pool_safe)
    