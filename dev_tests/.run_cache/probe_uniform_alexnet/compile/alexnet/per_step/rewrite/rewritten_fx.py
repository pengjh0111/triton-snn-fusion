


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
    getitem = l_x_seq_[0]
    input_1 = torch.conv2d(getitem, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem = None
    input_2 = torch.nn.functional.batch_norm(input_1, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_1 = None
    zeros_like = torch.zeros_like(input_2)
    lif_forward_state_default = torch.ops.snn_custom.lif_forward_state.default(input_2, zeros_like, 1.0, 0.0, 2.0, False);  input_2 = zeros_like = None
    spike = lif_forward_state_default[0]
    v_next = lif_forward_state_default[1];  lif_forward_state_default = None
    input_3 = torch.nn.functional.max_pool2d(spike, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike = None
    input_4 = torch.conv2d(input_3, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_3 = None
    input_5 = torch.nn.functional.batch_norm(input_4, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_4 = None
    zeros_like_1 = torch.zeros_like(input_5)
    lif_forward_state_default_1 = torch.ops.snn_custom.lif_forward_state.default(input_5, zeros_like_1, 1.0, 0.0, 2.0, False);  input_5 = zeros_like_1 = None
    spike_1 = lif_forward_state_default_1[0]
    v_next_1 = lif_forward_state_default_1[1];  lif_forward_state_default_1 = None
    input_6 = torch.nn.functional.max_pool2d(spike_1, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_1 = None
    input_7 = torch.conv2d(input_6, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_6 = None
    input_8 = torch.nn.functional.batch_norm(input_7, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_7 = None
    zeros_like_2 = torch.zeros_like(input_8)
    lif_forward_state_default_2 = torch.ops.snn_custom.lif_forward_state.default(input_8, zeros_like_2, 1.0, 0.0, 2.0, False);  input_8 = zeros_like_2 = None
    spike_2 = lif_forward_state_default_2[0]
    v_next_2 = lif_forward_state_default_2[1];  lif_forward_state_default_2 = None
    input_9 = torch.conv2d(spike_2, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_2 = None
    input_10 = torch.nn.functional.batch_norm(input_9, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_9 = None
    zeros_like_3 = torch.zeros_like(input_10)
    lif_forward_state_default_3 = torch.ops.snn_custom.lif_forward_state.default(input_10, zeros_like_3, 1.0, 0.0, 2.0, False);  input_10 = zeros_like_3 = None
    spike_3 = lif_forward_state_default_3[0]
    v_next_3 = lif_forward_state_default_3[1];  lif_forward_state_default_3 = None
    input_11 = torch.conv2d(spike_3, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_3 = None
    input_12 = torch.nn.functional.batch_norm(input_11, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_11 = None
    zeros_like_4 = torch.zeros_like(input_12)
    lif_forward_state_default_4 = torch.ops.snn_custom.lif_forward_state.default(input_12, zeros_like_4, 1.0, 0.0, 2.0, False);  input_12 = zeros_like_4 = None
    spike_4 = lif_forward_state_default_4[0]
    v_next_4 = lif_forward_state_default_4[1];  lif_forward_state_default_4 = None
    input_13 = torch.nn.functional.max_pool2d(spike_4, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_4 = None
    x = torch.nn.functional.adaptive_avg_pool2d(input_13, (6, 6));  input_13 = None
    x_1 = x.flatten(1, -1);  x = None
    input_14 = torch.nn.functional.dropout(x_1, 0.5, False, False);  x_1 = None
    input_15 = torch._C._nn.linear(input_14, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_14 = None
    zeros_like_5 = torch.zeros_like(input_15)
    lif_forward_state_default_5 = torch.ops.snn_custom.lif_forward_state.default(input_15, zeros_like_5, 1.0, 0.0, 2.0, False);  input_15 = zeros_like_5 = None
    spike_5 = lif_forward_state_default_5[0]
    v_next_5 = lif_forward_state_default_5[1];  lif_forward_state_default_5 = None
    input_16 = torch.nn.functional.dropout(spike_5, 0.5, False, False);  spike_5 = None
    input_17 = torch._C._nn.linear(input_16, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_16 = None
    zeros_like_6 = torch.zeros_like(input_17)
    lif_forward_state_default_6 = torch.ops.snn_custom.lif_forward_state.default(input_17, zeros_like_6, 1.0, 0.0, 2.0, False);  input_17 = zeros_like_6 = None
    spike_6 = lif_forward_state_default_6[0]
    v_next_6 = lif_forward_state_default_6[1];  lif_forward_state_default_6 = None
    input_18 = torch._C._nn.linear(spike_6, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_6 = None
    zeros_like_7 = torch.zeros_like(input_18)
    lif_forward_state_default_7 = torch.ops.snn_custom.lif_forward_state.default(input_18, zeros_like_7, 1.0, 0.0, 2.0, False);  input_18 = zeros_like_7 = None
    spike_7 = lif_forward_state_default_7[0]
    v_next_7 = lif_forward_state_default_7[1];  lif_forward_state_default_7 = None
    out_spikes_counter = 0 + spike_7;  spike_7 = None
    getitem_17 = l_x_seq_[1]
    input_19 = torch.conv2d(getitem_17, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_17 = None
    input_20 = torch.nn.functional.batch_norm(input_19, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_19 = None
    lif_forward_state_default_8 = torch.ops.snn_custom.lif_forward_state.default(input_20, v_next, 1.0, 0.0, 2.0, False);  input_20 = v_next = None
    spike_8 = lif_forward_state_default_8[0]
    v_next_8 = lif_forward_state_default_8[1];  lif_forward_state_default_8 = None
    input_21 = torch.nn.functional.max_pool2d(spike_8, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_8 = None
    input_22 = torch.conv2d(input_21, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_21 = None
    input_23 = torch.nn.functional.batch_norm(input_22, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_22 = None
    lif_forward_state_default_9 = torch.ops.snn_custom.lif_forward_state.default(input_23, v_next_1, 1.0, 0.0, 2.0, False);  input_23 = v_next_1 = None
    spike_9 = lif_forward_state_default_9[0]
    v_next_9 = lif_forward_state_default_9[1];  lif_forward_state_default_9 = None
    input_24 = torch.nn.functional.max_pool2d(spike_9, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_9 = None
    input_25 = torch.conv2d(input_24, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_24 = None
    input_26 = torch.nn.functional.batch_norm(input_25, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_25 = None
    lif_forward_state_default_10 = torch.ops.snn_custom.lif_forward_state.default(input_26, v_next_2, 1.0, 0.0, 2.0, False);  input_26 = v_next_2 = None
    spike_10 = lif_forward_state_default_10[0]
    v_next_10 = lif_forward_state_default_10[1];  lif_forward_state_default_10 = None
    input_27 = torch.conv2d(spike_10, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_10 = None
    input_28 = torch.nn.functional.batch_norm(input_27, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_27 = None
    lif_forward_state_default_11 = torch.ops.snn_custom.lif_forward_state.default(input_28, v_next_3, 1.0, 0.0, 2.0, False);  input_28 = v_next_3 = None
    spike_11 = lif_forward_state_default_11[0]
    v_next_11 = lif_forward_state_default_11[1];  lif_forward_state_default_11 = None
    input_29 = torch.conv2d(spike_11, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_11 = None
    input_30 = torch.nn.functional.batch_norm(input_29, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_29 = None
    lif_forward_state_default_12 = torch.ops.snn_custom.lif_forward_state.default(input_30, v_next_4, 1.0, 0.0, 2.0, False);  input_30 = v_next_4 = None
    spike_12 = lif_forward_state_default_12[0]
    v_next_12 = lif_forward_state_default_12[1];  lif_forward_state_default_12 = None
    input_31 = torch.nn.functional.max_pool2d(spike_12, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_12 = None
    x_2 = torch.nn.functional.adaptive_avg_pool2d(input_31, (6, 6));  input_31 = None
    x_3 = x_2.flatten(1, -1);  x_2 = None
    input_32 = torch.nn.functional.dropout(x_3, 0.5, False, False);  x_3 = None
    input_33 = torch._C._nn.linear(input_32, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_32 = None
    lif_forward_state_default_13 = torch.ops.snn_custom.lif_forward_state.default(input_33, v_next_5, 1.0, 0.0, 2.0, False);  input_33 = v_next_5 = None
    spike_13 = lif_forward_state_default_13[0]
    v_next_13 = lif_forward_state_default_13[1];  lif_forward_state_default_13 = None
    input_34 = torch.nn.functional.dropout(spike_13, 0.5, False, False);  spike_13 = None
    input_35 = torch._C._nn.linear(input_34, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_34 = None
    lif_forward_state_default_14 = torch.ops.snn_custom.lif_forward_state.default(input_35, v_next_6, 1.0, 0.0, 2.0, False);  input_35 = v_next_6 = None
    spike_14 = lif_forward_state_default_14[0]
    v_next_14 = lif_forward_state_default_14[1];  lif_forward_state_default_14 = None
    input_36 = torch._C._nn.linear(spike_14, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_14 = None
    lif_forward_state_default_15 = torch.ops.snn_custom.lif_forward_state.default(input_36, v_next_7, 1.0, 0.0, 2.0, False);  input_36 = v_next_7 = None
    spike_15 = lif_forward_state_default_15[0]
    v_next_15 = lif_forward_state_default_15[1];  lif_forward_state_default_15 = None
    out_spikes_counter_1 = out_spikes_counter + spike_15;  out_spikes_counter = spike_15 = None
    getitem_34 = l_x_seq_[2]
    input_37 = torch.conv2d(getitem_34, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_34 = None
    input_38 = torch.nn.functional.batch_norm(input_37, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_37 = None
    lif_forward_state_default_16 = torch.ops.snn_custom.lif_forward_state.default(input_38, v_next_8, 1.0, 0.0, 2.0, False);  input_38 = v_next_8 = None
    spike_16 = lif_forward_state_default_16[0]
    v_next_16 = lif_forward_state_default_16[1];  lif_forward_state_default_16 = None
    input_39 = torch.nn.functional.max_pool2d(spike_16, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_16 = None
    input_40 = torch.conv2d(input_39, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_39 = None
    input_41 = torch.nn.functional.batch_norm(input_40, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_40 = None
    lif_forward_state_default_17 = torch.ops.snn_custom.lif_forward_state.default(input_41, v_next_9, 1.0, 0.0, 2.0, False);  input_41 = v_next_9 = None
    spike_17 = lif_forward_state_default_17[0]
    v_next_17 = lif_forward_state_default_17[1];  lif_forward_state_default_17 = None
    input_42 = torch.nn.functional.max_pool2d(spike_17, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_17 = None
    input_43 = torch.conv2d(input_42, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_42 = None
    input_44 = torch.nn.functional.batch_norm(input_43, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_43 = None
    lif_forward_state_default_18 = torch.ops.snn_custom.lif_forward_state.default(input_44, v_next_10, 1.0, 0.0, 2.0, False);  input_44 = v_next_10 = None
    spike_18 = lif_forward_state_default_18[0]
    v_next_18 = lif_forward_state_default_18[1];  lif_forward_state_default_18 = None
    input_45 = torch.conv2d(spike_18, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_18 = None
    input_46 = torch.nn.functional.batch_norm(input_45, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_45 = None
    lif_forward_state_default_19 = torch.ops.snn_custom.lif_forward_state.default(input_46, v_next_11, 1.0, 0.0, 2.0, False);  input_46 = v_next_11 = None
    spike_19 = lif_forward_state_default_19[0]
    v_next_19 = lif_forward_state_default_19[1];  lif_forward_state_default_19 = None
    input_47 = torch.conv2d(spike_19, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_19 = None
    input_48 = torch.nn.functional.batch_norm(input_47, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_47 = None
    lif_forward_state_default_20 = torch.ops.snn_custom.lif_forward_state.default(input_48, v_next_12, 1.0, 0.0, 2.0, False);  input_48 = v_next_12 = None
    spike_20 = lif_forward_state_default_20[0]
    v_next_20 = lif_forward_state_default_20[1];  lif_forward_state_default_20 = None
    input_49 = torch.nn.functional.max_pool2d(spike_20, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_20 = None
    x_4 = torch.nn.functional.adaptive_avg_pool2d(input_49, (6, 6));  input_49 = None
    x_5 = x_4.flatten(1, -1);  x_4 = None
    input_50 = torch.nn.functional.dropout(x_5, 0.5, False, False);  x_5 = None
    input_51 = torch._C._nn.linear(input_50, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_50 = None
    lif_forward_state_default_21 = torch.ops.snn_custom.lif_forward_state.default(input_51, v_next_13, 1.0, 0.0, 2.0, False);  input_51 = v_next_13 = None
    spike_21 = lif_forward_state_default_21[0]
    v_next_21 = lif_forward_state_default_21[1];  lif_forward_state_default_21 = None
    input_52 = torch.nn.functional.dropout(spike_21, 0.5, False, False);  spike_21 = None
    input_53 = torch._C._nn.linear(input_52, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_52 = None
    lif_forward_state_default_22 = torch.ops.snn_custom.lif_forward_state.default(input_53, v_next_14, 1.0, 0.0, 2.0, False);  input_53 = v_next_14 = None
    spike_22 = lif_forward_state_default_22[0]
    v_next_22 = lif_forward_state_default_22[1];  lif_forward_state_default_22 = None
    input_54 = torch._C._nn.linear(spike_22, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_22 = None
    lif_forward_state_default_23 = torch.ops.snn_custom.lif_forward_state.default(input_54, v_next_15, 1.0, 0.0, 2.0, False);  input_54 = v_next_15 = None
    spike_23 = lif_forward_state_default_23[0]
    v_next_23 = lif_forward_state_default_23[1];  lif_forward_state_default_23 = None
    out_spikes_counter_2 = out_spikes_counter_1 + spike_23;  out_spikes_counter_1 = spike_23 = None
    getitem_51 = l_x_seq_[3]
    input_55 = torch.conv2d(getitem_51, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_51 = None
    input_56 = torch.nn.functional.batch_norm(input_55, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_55 = None
    lif_forward_state_default_24 = torch.ops.snn_custom.lif_forward_state.default(input_56, v_next_16, 1.0, 0.0, 2.0, False);  input_56 = v_next_16 = None
    spike_24 = lif_forward_state_default_24[0]
    v_next_24 = lif_forward_state_default_24[1];  lif_forward_state_default_24 = None
    input_57 = torch.nn.functional.max_pool2d(spike_24, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_24 = None
    input_58 = torch.conv2d(input_57, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_57 = None
    input_59 = torch.nn.functional.batch_norm(input_58, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_58 = None
    lif_forward_state_default_25 = torch.ops.snn_custom.lif_forward_state.default(input_59, v_next_17, 1.0, 0.0, 2.0, False);  input_59 = v_next_17 = None
    spike_25 = lif_forward_state_default_25[0]
    v_next_25 = lif_forward_state_default_25[1];  lif_forward_state_default_25 = None
    input_60 = torch.nn.functional.max_pool2d(spike_25, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_25 = None
    input_61 = torch.conv2d(input_60, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_60 = None
    input_62 = torch.nn.functional.batch_norm(input_61, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_61 = None
    lif_forward_state_default_26 = torch.ops.snn_custom.lif_forward_state.default(input_62, v_next_18, 1.0, 0.0, 2.0, False);  input_62 = v_next_18 = None
    spike_26 = lif_forward_state_default_26[0]
    v_next_26 = lif_forward_state_default_26[1];  lif_forward_state_default_26 = None
    input_63 = torch.conv2d(spike_26, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_26 = None
    input_64 = torch.nn.functional.batch_norm(input_63, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_63 = None
    lif_forward_state_default_27 = torch.ops.snn_custom.lif_forward_state.default(input_64, v_next_19, 1.0, 0.0, 2.0, False);  input_64 = v_next_19 = None
    spike_27 = lif_forward_state_default_27[0]
    v_next_27 = lif_forward_state_default_27[1];  lif_forward_state_default_27 = None
    input_65 = torch.conv2d(spike_27, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_27 = None
    input_66 = torch.nn.functional.batch_norm(input_65, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_65 = None
    lif_forward_state_default_28 = torch.ops.snn_custom.lif_forward_state.default(input_66, v_next_20, 1.0, 0.0, 2.0, False);  input_66 = v_next_20 = None
    spike_28 = lif_forward_state_default_28[0]
    v_next_28 = lif_forward_state_default_28[1];  lif_forward_state_default_28 = None
    input_67 = torch.nn.functional.max_pool2d(spike_28, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_28 = None
    x_6 = torch.nn.functional.adaptive_avg_pool2d(input_67, (6, 6));  input_67 = None
    x_7 = x_6.flatten(1, -1);  x_6 = None
    input_68 = torch.nn.functional.dropout(x_7, 0.5, False, False);  x_7 = None
    input_69 = torch._C._nn.linear(input_68, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_68 = None
    lif_forward_state_default_29 = torch.ops.snn_custom.lif_forward_state.default(input_69, v_next_21, 1.0, 0.0, 2.0, False);  input_69 = v_next_21 = None
    spike_29 = lif_forward_state_default_29[0]
    v_next_29 = lif_forward_state_default_29[1];  lif_forward_state_default_29 = None
    input_70 = torch.nn.functional.dropout(spike_29, 0.5, False, False);  spike_29 = None
    input_71 = torch._C._nn.linear(input_70, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_70 = None
    lif_forward_state_default_30 = torch.ops.snn_custom.lif_forward_state.default(input_71, v_next_22, 1.0, 0.0, 2.0, False);  input_71 = v_next_22 = None
    spike_30 = lif_forward_state_default_30[0]
    v_next_30 = lif_forward_state_default_30[1];  lif_forward_state_default_30 = None
    input_72 = torch._C._nn.linear(spike_30, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_30 = None
    lif_forward_state_default_31 = torch.ops.snn_custom.lif_forward_state.default(input_72, v_next_23, 1.0, 0.0, 2.0, False);  input_72 = v_next_23 = None
    spike_31 = lif_forward_state_default_31[0]
    v_next_31 = lif_forward_state_default_31[1];  lif_forward_state_default_31 = None
    out_spikes_counter_3 = out_spikes_counter_2 + spike_31;  out_spikes_counter_2 = spike_31 = None
    getitem_68 = l_x_seq_[4]
    input_73 = torch.conv2d(getitem_68, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_68 = None
    input_74 = torch.nn.functional.batch_norm(input_73, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_73 = None
    lif_forward_state_default_32 = torch.ops.snn_custom.lif_forward_state.default(input_74, v_next_24, 1.0, 0.0, 2.0, False);  input_74 = v_next_24 = None
    spike_32 = lif_forward_state_default_32[0]
    v_next_32 = lif_forward_state_default_32[1];  lif_forward_state_default_32 = None
    input_75 = torch.nn.functional.max_pool2d(spike_32, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_32 = None
    input_76 = torch.conv2d(input_75, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_75 = None
    input_77 = torch.nn.functional.batch_norm(input_76, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_76 = None
    lif_forward_state_default_33 = torch.ops.snn_custom.lif_forward_state.default(input_77, v_next_25, 1.0, 0.0, 2.0, False);  input_77 = v_next_25 = None
    spike_33 = lif_forward_state_default_33[0]
    v_next_33 = lif_forward_state_default_33[1];  lif_forward_state_default_33 = None
    input_78 = torch.nn.functional.max_pool2d(spike_33, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_33 = None
    input_79 = torch.conv2d(input_78, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_78 = None
    input_80 = torch.nn.functional.batch_norm(input_79, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_79 = None
    lif_forward_state_default_34 = torch.ops.snn_custom.lif_forward_state.default(input_80, v_next_26, 1.0, 0.0, 2.0, False);  input_80 = v_next_26 = None
    spike_34 = lif_forward_state_default_34[0]
    v_next_34 = lif_forward_state_default_34[1];  lif_forward_state_default_34 = None
    input_81 = torch.conv2d(spike_34, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_34 = None
    input_82 = torch.nn.functional.batch_norm(input_81, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_81 = None
    lif_forward_state_default_35 = torch.ops.snn_custom.lif_forward_state.default(input_82, v_next_27, 1.0, 0.0, 2.0, False);  input_82 = v_next_27 = None
    spike_35 = lif_forward_state_default_35[0]
    v_next_35 = lif_forward_state_default_35[1];  lif_forward_state_default_35 = None
    input_83 = torch.conv2d(spike_35, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_35 = None
    input_84 = torch.nn.functional.batch_norm(input_83, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_83 = None
    lif_forward_state_default_36 = torch.ops.snn_custom.lif_forward_state.default(input_84, v_next_28, 1.0, 0.0, 2.0, False);  input_84 = v_next_28 = None
    spike_36 = lif_forward_state_default_36[0]
    v_next_36 = lif_forward_state_default_36[1];  lif_forward_state_default_36 = None
    input_85 = torch.nn.functional.max_pool2d(spike_36, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_36 = None
    x_8 = torch.nn.functional.adaptive_avg_pool2d(input_85, (6, 6));  input_85 = None
    x_9 = x_8.flatten(1, -1);  x_8 = None
    input_86 = torch.nn.functional.dropout(x_9, 0.5, False, False);  x_9 = None
    input_87 = torch._C._nn.linear(input_86, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_86 = None
    lif_forward_state_default_37 = torch.ops.snn_custom.lif_forward_state.default(input_87, v_next_29, 1.0, 0.0, 2.0, False);  input_87 = v_next_29 = None
    spike_37 = lif_forward_state_default_37[0]
    v_next_37 = lif_forward_state_default_37[1];  lif_forward_state_default_37 = None
    input_88 = torch.nn.functional.dropout(spike_37, 0.5, False, False);  spike_37 = None
    input_89 = torch._C._nn.linear(input_88, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_88 = None
    lif_forward_state_default_38 = torch.ops.snn_custom.lif_forward_state.default(input_89, v_next_30, 1.0, 0.0, 2.0, False);  input_89 = v_next_30 = None
    spike_38 = lif_forward_state_default_38[0]
    v_next_38 = lif_forward_state_default_38[1];  lif_forward_state_default_38 = None
    input_90 = torch._C._nn.linear(spike_38, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_38 = None
    lif_forward_state_default_39 = torch.ops.snn_custom.lif_forward_state.default(input_90, v_next_31, 1.0, 0.0, 2.0, False);  input_90 = v_next_31 = None
    spike_39 = lif_forward_state_default_39[0]
    v_next_39 = lif_forward_state_default_39[1];  lif_forward_state_default_39 = None
    out_spikes_counter_4 = out_spikes_counter_3 + spike_39;  out_spikes_counter_3 = spike_39 = None
    getitem_85 = l_x_seq_[5]
    input_91 = torch.conv2d(getitem_85, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_85 = None
    input_92 = torch.nn.functional.batch_norm(input_91, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_91 = None
    lif_forward_state_default_40 = torch.ops.snn_custom.lif_forward_state.default(input_92, v_next_32, 1.0, 0.0, 2.0, False);  input_92 = v_next_32 = None
    spike_40 = lif_forward_state_default_40[0]
    v_next_40 = lif_forward_state_default_40[1];  lif_forward_state_default_40 = None
    input_93 = torch.nn.functional.max_pool2d(spike_40, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_40 = None
    input_94 = torch.conv2d(input_93, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_93 = None
    input_95 = torch.nn.functional.batch_norm(input_94, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_94 = None
    lif_forward_state_default_41 = torch.ops.snn_custom.lif_forward_state.default(input_95, v_next_33, 1.0, 0.0, 2.0, False);  input_95 = v_next_33 = None
    spike_41 = lif_forward_state_default_41[0]
    v_next_41 = lif_forward_state_default_41[1];  lif_forward_state_default_41 = None
    input_96 = torch.nn.functional.max_pool2d(spike_41, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_41 = None
    input_97 = torch.conv2d(input_96, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_96 = None
    input_98 = torch.nn.functional.batch_norm(input_97, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_97 = None
    lif_forward_state_default_42 = torch.ops.snn_custom.lif_forward_state.default(input_98, v_next_34, 1.0, 0.0, 2.0, False);  input_98 = v_next_34 = None
    spike_42 = lif_forward_state_default_42[0]
    v_next_42 = lif_forward_state_default_42[1];  lif_forward_state_default_42 = None
    input_99 = torch.conv2d(spike_42, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_42 = None
    input_100 = torch.nn.functional.batch_norm(input_99, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_99 = None
    lif_forward_state_default_43 = torch.ops.snn_custom.lif_forward_state.default(input_100, v_next_35, 1.0, 0.0, 2.0, False);  input_100 = v_next_35 = None
    spike_43 = lif_forward_state_default_43[0]
    v_next_43 = lif_forward_state_default_43[1];  lif_forward_state_default_43 = None
    input_101 = torch.conv2d(spike_43, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_43 = None
    input_102 = torch.nn.functional.batch_norm(input_101, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_101 = None
    lif_forward_state_default_44 = torch.ops.snn_custom.lif_forward_state.default(input_102, v_next_36, 1.0, 0.0, 2.0, False);  input_102 = v_next_36 = None
    spike_44 = lif_forward_state_default_44[0]
    v_next_44 = lif_forward_state_default_44[1];  lif_forward_state_default_44 = None
    input_103 = torch.nn.functional.max_pool2d(spike_44, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_44 = None
    x_10 = torch.nn.functional.adaptive_avg_pool2d(input_103, (6, 6));  input_103 = None
    x_11 = x_10.flatten(1, -1);  x_10 = None
    input_104 = torch.nn.functional.dropout(x_11, 0.5, False, False);  x_11 = None
    input_105 = torch._C._nn.linear(input_104, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_104 = None
    lif_forward_state_default_45 = torch.ops.snn_custom.lif_forward_state.default(input_105, v_next_37, 1.0, 0.0, 2.0, False);  input_105 = v_next_37 = None
    spike_45 = lif_forward_state_default_45[0]
    v_next_45 = lif_forward_state_default_45[1];  lif_forward_state_default_45 = None
    input_106 = torch.nn.functional.dropout(spike_45, 0.5, False, False);  spike_45 = None
    input_107 = torch._C._nn.linear(input_106, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_106 = None
    lif_forward_state_default_46 = torch.ops.snn_custom.lif_forward_state.default(input_107, v_next_38, 1.0, 0.0, 2.0, False);  input_107 = v_next_38 = None
    spike_46 = lif_forward_state_default_46[0]
    v_next_46 = lif_forward_state_default_46[1];  lif_forward_state_default_46 = None
    input_108 = torch._C._nn.linear(spike_46, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_46 = None
    lif_forward_state_default_47 = torch.ops.snn_custom.lif_forward_state.default(input_108, v_next_39, 1.0, 0.0, 2.0, False);  input_108 = v_next_39 = None
    spike_47 = lif_forward_state_default_47[0]
    v_next_47 = lif_forward_state_default_47[1];  lif_forward_state_default_47 = None
    out_spikes_counter_5 = out_spikes_counter_4 + spike_47;  out_spikes_counter_4 = spike_47 = None
    getitem_102 = l_x_seq_[6]
    input_109 = torch.conv2d(getitem_102, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_102 = None
    input_110 = torch.nn.functional.batch_norm(input_109, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_109 = None
    lif_forward_state_default_48 = torch.ops.snn_custom.lif_forward_state.default(input_110, v_next_40, 1.0, 0.0, 2.0, False);  input_110 = v_next_40 = None
    spike_48 = lif_forward_state_default_48[0]
    v_next_48 = lif_forward_state_default_48[1];  lif_forward_state_default_48 = None
    input_111 = torch.nn.functional.max_pool2d(spike_48, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_48 = None
    input_112 = torch.conv2d(input_111, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_111 = None
    input_113 = torch.nn.functional.batch_norm(input_112, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_112 = None
    lif_forward_state_default_49 = torch.ops.snn_custom.lif_forward_state.default(input_113, v_next_41, 1.0, 0.0, 2.0, False);  input_113 = v_next_41 = None
    spike_49 = lif_forward_state_default_49[0]
    v_next_49 = lif_forward_state_default_49[1];  lif_forward_state_default_49 = None
    input_114 = torch.nn.functional.max_pool2d(spike_49, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_49 = None
    input_115 = torch.conv2d(input_114, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_114 = None
    input_116 = torch.nn.functional.batch_norm(input_115, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_115 = None
    lif_forward_state_default_50 = torch.ops.snn_custom.lif_forward_state.default(input_116, v_next_42, 1.0, 0.0, 2.0, False);  input_116 = v_next_42 = None
    spike_50 = lif_forward_state_default_50[0]
    v_next_50 = lif_forward_state_default_50[1];  lif_forward_state_default_50 = None
    input_117 = torch.conv2d(spike_50, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_50 = None
    input_118 = torch.nn.functional.batch_norm(input_117, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_117 = None
    lif_forward_state_default_51 = torch.ops.snn_custom.lif_forward_state.default(input_118, v_next_43, 1.0, 0.0, 2.0, False);  input_118 = v_next_43 = None
    spike_51 = lif_forward_state_default_51[0]
    v_next_51 = lif_forward_state_default_51[1];  lif_forward_state_default_51 = None
    input_119 = torch.conv2d(spike_51, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_51 = None
    input_120 = torch.nn.functional.batch_norm(input_119, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_119 = None
    lif_forward_state_default_52 = torch.ops.snn_custom.lif_forward_state.default(input_120, v_next_44, 1.0, 0.0, 2.0, False);  input_120 = v_next_44 = None
    spike_52 = lif_forward_state_default_52[0]
    v_next_52 = lif_forward_state_default_52[1];  lif_forward_state_default_52 = None
    input_121 = torch.nn.functional.max_pool2d(spike_52, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_52 = None
    x_12 = torch.nn.functional.adaptive_avg_pool2d(input_121, (6, 6));  input_121 = None
    x_13 = x_12.flatten(1, -1);  x_12 = None
    input_122 = torch.nn.functional.dropout(x_13, 0.5, False, False);  x_13 = None
    input_123 = torch._C._nn.linear(input_122, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_122 = None
    lif_forward_state_default_53 = torch.ops.snn_custom.lif_forward_state.default(input_123, v_next_45, 1.0, 0.0, 2.0, False);  input_123 = v_next_45 = None
    spike_53 = lif_forward_state_default_53[0]
    v_next_53 = lif_forward_state_default_53[1];  lif_forward_state_default_53 = None
    input_124 = torch.nn.functional.dropout(spike_53, 0.5, False, False);  spike_53 = None
    input_125 = torch._C._nn.linear(input_124, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_124 = None
    lif_forward_state_default_54 = torch.ops.snn_custom.lif_forward_state.default(input_125, v_next_46, 1.0, 0.0, 2.0, False);  input_125 = v_next_46 = None
    spike_54 = lif_forward_state_default_54[0]
    v_next_54 = lif_forward_state_default_54[1];  lif_forward_state_default_54 = None
    input_126 = torch._C._nn.linear(spike_54, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_54 = None
    lif_forward_state_default_55 = torch.ops.snn_custom.lif_forward_state.default(input_126, v_next_47, 1.0, 0.0, 2.0, False);  input_126 = v_next_47 = None
    spike_55 = lif_forward_state_default_55[0]
    v_next_55 = lif_forward_state_default_55[1];  lif_forward_state_default_55 = None
    out_spikes_counter_6 = out_spikes_counter_5 + spike_55;  out_spikes_counter_5 = spike_55 = None
    getitem_119 = l_x_seq_[7]
    input_127 = torch.conv2d(getitem_119, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_119 = None
    input_128 = torch.nn.functional.batch_norm(input_127, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_127 = None
    lif_forward_state_default_56 = torch.ops.snn_custom.lif_forward_state.default(input_128, v_next_48, 1.0, 0.0, 2.0, False);  input_128 = v_next_48 = None
    spike_56 = lif_forward_state_default_56[0]
    v_next_56 = lif_forward_state_default_56[1];  lif_forward_state_default_56 = None
    input_129 = torch.nn.functional.max_pool2d(spike_56, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_56 = None
    input_130 = torch.conv2d(input_129, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_129 = None
    input_131 = torch.nn.functional.batch_norm(input_130, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_130 = None
    lif_forward_state_default_57 = torch.ops.snn_custom.lif_forward_state.default(input_131, v_next_49, 1.0, 0.0, 2.0, False);  input_131 = v_next_49 = None
    spike_57 = lif_forward_state_default_57[0]
    v_next_57 = lif_forward_state_default_57[1];  lif_forward_state_default_57 = None
    input_132 = torch.nn.functional.max_pool2d(spike_57, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_57 = None
    input_133 = torch.conv2d(input_132, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_132 = None
    input_134 = torch.nn.functional.batch_norm(input_133, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_133 = None
    lif_forward_state_default_58 = torch.ops.snn_custom.lif_forward_state.default(input_134, v_next_50, 1.0, 0.0, 2.0, False);  input_134 = v_next_50 = None
    spike_58 = lif_forward_state_default_58[0]
    v_next_58 = lif_forward_state_default_58[1];  lif_forward_state_default_58 = None
    input_135 = torch.conv2d(spike_58, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_58 = None
    input_136 = torch.nn.functional.batch_norm(input_135, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_135 = None
    lif_forward_state_default_59 = torch.ops.snn_custom.lif_forward_state.default(input_136, v_next_51, 1.0, 0.0, 2.0, False);  input_136 = v_next_51 = None
    spike_59 = lif_forward_state_default_59[0]
    v_next_59 = lif_forward_state_default_59[1];  lif_forward_state_default_59 = None
    input_137 = torch.conv2d(spike_59, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_59 = None
    input_138 = torch.nn.functional.batch_norm(input_137, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_137 = None
    lif_forward_state_default_60 = torch.ops.snn_custom.lif_forward_state.default(input_138, v_next_52, 1.0, 0.0, 2.0, False);  input_138 = v_next_52 = None
    spike_60 = lif_forward_state_default_60[0]
    v_next_60 = lif_forward_state_default_60[1];  lif_forward_state_default_60 = None
    input_139 = torch.nn.functional.max_pool2d(spike_60, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_60 = None
    x_14 = torch.nn.functional.adaptive_avg_pool2d(input_139, (6, 6));  input_139 = None
    x_15 = x_14.flatten(1, -1);  x_14 = None
    input_140 = torch.nn.functional.dropout(x_15, 0.5, False, False);  x_15 = None
    input_141 = torch._C._nn.linear(input_140, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_140 = None
    lif_forward_state_default_61 = torch.ops.snn_custom.lif_forward_state.default(input_141, v_next_53, 1.0, 0.0, 2.0, False);  input_141 = v_next_53 = None
    spike_61 = lif_forward_state_default_61[0]
    v_next_61 = lif_forward_state_default_61[1];  lif_forward_state_default_61 = None
    input_142 = torch.nn.functional.dropout(spike_61, 0.5, False, False);  spike_61 = None
    input_143 = torch._C._nn.linear(input_142, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_142 = None
    lif_forward_state_default_62 = torch.ops.snn_custom.lif_forward_state.default(input_143, v_next_54, 1.0, 0.0, 2.0, False);  input_143 = v_next_54 = None
    spike_62 = lif_forward_state_default_62[0]
    v_next_62 = lif_forward_state_default_62[1];  lif_forward_state_default_62 = None
    input_144 = torch._C._nn.linear(spike_62, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_62 = None
    lif_forward_state_default_63 = torch.ops.snn_custom.lif_forward_state.default(input_144, v_next_55, 1.0, 0.0, 2.0, False);  input_144 = v_next_55 = None
    spike_63 = lif_forward_state_default_63[0]
    v_next_63 = lif_forward_state_default_63[1];  lif_forward_state_default_63 = None
    out_spikes_counter_7 = out_spikes_counter_6 + spike_63;  out_spikes_counter_6 = spike_63 = None
    getitem_136 = l_x_seq_[8]
    input_145 = torch.conv2d(getitem_136, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_136 = None
    input_146 = torch.nn.functional.batch_norm(input_145, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_145 = None
    lif_forward_state_default_64 = torch.ops.snn_custom.lif_forward_state.default(input_146, v_next_56, 1.0, 0.0, 2.0, False);  input_146 = v_next_56 = None
    spike_64 = lif_forward_state_default_64[0]
    v_next_64 = lif_forward_state_default_64[1];  lif_forward_state_default_64 = None
    input_147 = torch.nn.functional.max_pool2d(spike_64, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_64 = None
    input_148 = torch.conv2d(input_147, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_147 = None
    input_149 = torch.nn.functional.batch_norm(input_148, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_148 = None
    lif_forward_state_default_65 = torch.ops.snn_custom.lif_forward_state.default(input_149, v_next_57, 1.0, 0.0, 2.0, False);  input_149 = v_next_57 = None
    spike_65 = lif_forward_state_default_65[0]
    v_next_65 = lif_forward_state_default_65[1];  lif_forward_state_default_65 = None
    input_150 = torch.nn.functional.max_pool2d(spike_65, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_65 = None
    input_151 = torch.conv2d(input_150, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_150 = None
    input_152 = torch.nn.functional.batch_norm(input_151, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_151 = None
    lif_forward_state_default_66 = torch.ops.snn_custom.lif_forward_state.default(input_152, v_next_58, 1.0, 0.0, 2.0, False);  input_152 = v_next_58 = None
    spike_66 = lif_forward_state_default_66[0]
    v_next_66 = lif_forward_state_default_66[1];  lif_forward_state_default_66 = None
    input_153 = torch.conv2d(spike_66, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_66 = None
    input_154 = torch.nn.functional.batch_norm(input_153, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_153 = None
    lif_forward_state_default_67 = torch.ops.snn_custom.lif_forward_state.default(input_154, v_next_59, 1.0, 0.0, 2.0, False);  input_154 = v_next_59 = None
    spike_67 = lif_forward_state_default_67[0]
    v_next_67 = lif_forward_state_default_67[1];  lif_forward_state_default_67 = None
    input_155 = torch.conv2d(spike_67, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_67 = None
    input_156 = torch.nn.functional.batch_norm(input_155, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_155 = None
    lif_forward_state_default_68 = torch.ops.snn_custom.lif_forward_state.default(input_156, v_next_60, 1.0, 0.0, 2.0, False);  input_156 = v_next_60 = None
    spike_68 = lif_forward_state_default_68[0]
    v_next_68 = lif_forward_state_default_68[1];  lif_forward_state_default_68 = None
    input_157 = torch.nn.functional.max_pool2d(spike_68, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_68 = None
    x_16 = torch.nn.functional.adaptive_avg_pool2d(input_157, (6, 6));  input_157 = None
    x_17 = x_16.flatten(1, -1);  x_16 = None
    input_158 = torch.nn.functional.dropout(x_17, 0.5, False, False);  x_17 = None
    input_159 = torch._C._nn.linear(input_158, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_158 = None
    lif_forward_state_default_69 = torch.ops.snn_custom.lif_forward_state.default(input_159, v_next_61, 1.0, 0.0, 2.0, False);  input_159 = v_next_61 = None
    spike_69 = lif_forward_state_default_69[0]
    v_next_69 = lif_forward_state_default_69[1];  lif_forward_state_default_69 = None
    input_160 = torch.nn.functional.dropout(spike_69, 0.5, False, False);  spike_69 = None
    input_161 = torch._C._nn.linear(input_160, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_160 = None
    lif_forward_state_default_70 = torch.ops.snn_custom.lif_forward_state.default(input_161, v_next_62, 1.0, 0.0, 2.0, False);  input_161 = v_next_62 = None
    spike_70 = lif_forward_state_default_70[0]
    v_next_70 = lif_forward_state_default_70[1];  lif_forward_state_default_70 = None
    input_162 = torch._C._nn.linear(spike_70, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_70 = None
    lif_forward_state_default_71 = torch.ops.snn_custom.lif_forward_state.default(input_162, v_next_63, 1.0, 0.0, 2.0, False);  input_162 = v_next_63 = None
    spike_71 = lif_forward_state_default_71[0]
    v_next_71 = lif_forward_state_default_71[1];  lif_forward_state_default_71 = None
    out_spikes_counter_8 = out_spikes_counter_7 + spike_71;  out_spikes_counter_7 = spike_71 = None
    getitem_153 = l_x_seq_[9]
    input_163 = torch.conv2d(getitem_153, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_153 = None
    input_164 = torch.nn.functional.batch_norm(input_163, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_163 = None
    lif_forward_state_default_72 = torch.ops.snn_custom.lif_forward_state.default(input_164, v_next_64, 1.0, 0.0, 2.0, False);  input_164 = v_next_64 = None
    spike_72 = lif_forward_state_default_72[0]
    v_next_72 = lif_forward_state_default_72[1];  lif_forward_state_default_72 = None
    input_165 = torch.nn.functional.max_pool2d(spike_72, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_72 = None
    input_166 = torch.conv2d(input_165, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_165 = None
    input_167 = torch.nn.functional.batch_norm(input_166, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_166 = None
    lif_forward_state_default_73 = torch.ops.snn_custom.lif_forward_state.default(input_167, v_next_65, 1.0, 0.0, 2.0, False);  input_167 = v_next_65 = None
    spike_73 = lif_forward_state_default_73[0]
    v_next_73 = lif_forward_state_default_73[1];  lif_forward_state_default_73 = None
    input_168 = torch.nn.functional.max_pool2d(spike_73, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_73 = None
    input_169 = torch.conv2d(input_168, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_168 = None
    input_170 = torch.nn.functional.batch_norm(input_169, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_169 = None
    lif_forward_state_default_74 = torch.ops.snn_custom.lif_forward_state.default(input_170, v_next_66, 1.0, 0.0, 2.0, False);  input_170 = v_next_66 = None
    spike_74 = lif_forward_state_default_74[0]
    v_next_74 = lif_forward_state_default_74[1];  lif_forward_state_default_74 = None
    input_171 = torch.conv2d(spike_74, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_74 = None
    input_172 = torch.nn.functional.batch_norm(input_171, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_171 = None
    lif_forward_state_default_75 = torch.ops.snn_custom.lif_forward_state.default(input_172, v_next_67, 1.0, 0.0, 2.0, False);  input_172 = v_next_67 = None
    spike_75 = lif_forward_state_default_75[0]
    v_next_75 = lif_forward_state_default_75[1];  lif_forward_state_default_75 = None
    input_173 = torch.conv2d(spike_75, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_75 = None
    input_174 = torch.nn.functional.batch_norm(input_173, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_173 = None
    lif_forward_state_default_76 = torch.ops.snn_custom.lif_forward_state.default(input_174, v_next_68, 1.0, 0.0, 2.0, False);  input_174 = v_next_68 = None
    spike_76 = lif_forward_state_default_76[0]
    v_next_76 = lif_forward_state_default_76[1];  lif_forward_state_default_76 = None
    input_175 = torch.nn.functional.max_pool2d(spike_76, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_76 = None
    x_18 = torch.nn.functional.adaptive_avg_pool2d(input_175, (6, 6));  input_175 = None
    x_19 = x_18.flatten(1, -1);  x_18 = None
    input_176 = torch.nn.functional.dropout(x_19, 0.5, False, False);  x_19 = None
    input_177 = torch._C._nn.linear(input_176, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_176 = None
    lif_forward_state_default_77 = torch.ops.snn_custom.lif_forward_state.default(input_177, v_next_69, 1.0, 0.0, 2.0, False);  input_177 = v_next_69 = None
    spike_77 = lif_forward_state_default_77[0]
    v_next_77 = lif_forward_state_default_77[1];  lif_forward_state_default_77 = None
    input_178 = torch.nn.functional.dropout(spike_77, 0.5, False, False);  spike_77 = None
    input_179 = torch._C._nn.linear(input_178, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_178 = None
    lif_forward_state_default_78 = torch.ops.snn_custom.lif_forward_state.default(input_179, v_next_70, 1.0, 0.0, 2.0, False);  input_179 = v_next_70 = None
    spike_78 = lif_forward_state_default_78[0]
    v_next_78 = lif_forward_state_default_78[1];  lif_forward_state_default_78 = None
    input_180 = torch._C._nn.linear(spike_78, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_78 = None
    lif_forward_state_default_79 = torch.ops.snn_custom.lif_forward_state.default(input_180, v_next_71, 1.0, 0.0, 2.0, False);  input_180 = v_next_71 = None
    spike_79 = lif_forward_state_default_79[0]
    v_next_79 = lif_forward_state_default_79[1];  lif_forward_state_default_79 = None
    out_spikes_counter_9 = out_spikes_counter_8 + spike_79;  out_spikes_counter_8 = spike_79 = None
    getitem_170 = l_x_seq_[10]
    input_181 = torch.conv2d(getitem_170, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_170 = None
    input_182 = torch.nn.functional.batch_norm(input_181, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_181 = None
    lif_forward_state_default_80 = torch.ops.snn_custom.lif_forward_state.default(input_182, v_next_72, 1.0, 0.0, 2.0, False);  input_182 = v_next_72 = None
    spike_80 = lif_forward_state_default_80[0]
    v_next_80 = lif_forward_state_default_80[1];  lif_forward_state_default_80 = None
    input_183 = torch.nn.functional.max_pool2d(spike_80, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_80 = None
    input_184 = torch.conv2d(input_183, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_183 = None
    input_185 = torch.nn.functional.batch_norm(input_184, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_184 = None
    lif_forward_state_default_81 = torch.ops.snn_custom.lif_forward_state.default(input_185, v_next_73, 1.0, 0.0, 2.0, False);  input_185 = v_next_73 = None
    spike_81 = lif_forward_state_default_81[0]
    v_next_81 = lif_forward_state_default_81[1];  lif_forward_state_default_81 = None
    input_186 = torch.nn.functional.max_pool2d(spike_81, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_81 = None
    input_187 = torch.conv2d(input_186, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_186 = None
    input_188 = torch.nn.functional.batch_norm(input_187, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_187 = None
    lif_forward_state_default_82 = torch.ops.snn_custom.lif_forward_state.default(input_188, v_next_74, 1.0, 0.0, 2.0, False);  input_188 = v_next_74 = None
    spike_82 = lif_forward_state_default_82[0]
    v_next_82 = lif_forward_state_default_82[1];  lif_forward_state_default_82 = None
    input_189 = torch.conv2d(spike_82, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_82 = None
    input_190 = torch.nn.functional.batch_norm(input_189, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_189 = None
    lif_forward_state_default_83 = torch.ops.snn_custom.lif_forward_state.default(input_190, v_next_75, 1.0, 0.0, 2.0, False);  input_190 = v_next_75 = None
    spike_83 = lif_forward_state_default_83[0]
    v_next_83 = lif_forward_state_default_83[1];  lif_forward_state_default_83 = None
    input_191 = torch.conv2d(spike_83, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_83 = None
    input_192 = torch.nn.functional.batch_norm(input_191, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_191 = None
    lif_forward_state_default_84 = torch.ops.snn_custom.lif_forward_state.default(input_192, v_next_76, 1.0, 0.0, 2.0, False);  input_192 = v_next_76 = None
    spike_84 = lif_forward_state_default_84[0]
    v_next_84 = lif_forward_state_default_84[1];  lif_forward_state_default_84 = None
    input_193 = torch.nn.functional.max_pool2d(spike_84, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_84 = None
    x_20 = torch.nn.functional.adaptive_avg_pool2d(input_193, (6, 6));  input_193 = None
    x_21 = x_20.flatten(1, -1);  x_20 = None
    input_194 = torch.nn.functional.dropout(x_21, 0.5, False, False);  x_21 = None
    input_195 = torch._C._nn.linear(input_194, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_194 = None
    lif_forward_state_default_85 = torch.ops.snn_custom.lif_forward_state.default(input_195, v_next_77, 1.0, 0.0, 2.0, False);  input_195 = v_next_77 = None
    spike_85 = lif_forward_state_default_85[0]
    v_next_85 = lif_forward_state_default_85[1];  lif_forward_state_default_85 = None
    input_196 = torch.nn.functional.dropout(spike_85, 0.5, False, False);  spike_85 = None
    input_197 = torch._C._nn.linear(input_196, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_196 = None
    lif_forward_state_default_86 = torch.ops.snn_custom.lif_forward_state.default(input_197, v_next_78, 1.0, 0.0, 2.0, False);  input_197 = v_next_78 = None
    spike_86 = lif_forward_state_default_86[0]
    v_next_86 = lif_forward_state_default_86[1];  lif_forward_state_default_86 = None
    input_198 = torch._C._nn.linear(spike_86, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_86 = None
    lif_forward_state_default_87 = torch.ops.snn_custom.lif_forward_state.default(input_198, v_next_79, 1.0, 0.0, 2.0, False);  input_198 = v_next_79 = None
    spike_87 = lif_forward_state_default_87[0]
    v_next_87 = lif_forward_state_default_87[1];  lif_forward_state_default_87 = None
    out_spikes_counter_10 = out_spikes_counter_9 + spike_87;  out_spikes_counter_9 = spike_87 = None
    getitem_187 = l_x_seq_[11]
    input_199 = torch.conv2d(getitem_187, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_187 = None
    input_200 = torch.nn.functional.batch_norm(input_199, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_199 = None
    lif_forward_state_default_88 = torch.ops.snn_custom.lif_forward_state.default(input_200, v_next_80, 1.0, 0.0, 2.0, False);  input_200 = v_next_80 = None
    spike_88 = lif_forward_state_default_88[0]
    v_next_88 = lif_forward_state_default_88[1];  lif_forward_state_default_88 = None
    input_201 = torch.nn.functional.max_pool2d(spike_88, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_88 = None
    input_202 = torch.conv2d(input_201, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_201 = None
    input_203 = torch.nn.functional.batch_norm(input_202, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_202 = None
    lif_forward_state_default_89 = torch.ops.snn_custom.lif_forward_state.default(input_203, v_next_81, 1.0, 0.0, 2.0, False);  input_203 = v_next_81 = None
    spike_89 = lif_forward_state_default_89[0]
    v_next_89 = lif_forward_state_default_89[1];  lif_forward_state_default_89 = None
    input_204 = torch.nn.functional.max_pool2d(spike_89, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_89 = None
    input_205 = torch.conv2d(input_204, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_204 = None
    input_206 = torch.nn.functional.batch_norm(input_205, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_205 = None
    lif_forward_state_default_90 = torch.ops.snn_custom.lif_forward_state.default(input_206, v_next_82, 1.0, 0.0, 2.0, False);  input_206 = v_next_82 = None
    spike_90 = lif_forward_state_default_90[0]
    v_next_90 = lif_forward_state_default_90[1];  lif_forward_state_default_90 = None
    input_207 = torch.conv2d(spike_90, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_90 = None
    input_208 = torch.nn.functional.batch_norm(input_207, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_207 = None
    lif_forward_state_default_91 = torch.ops.snn_custom.lif_forward_state.default(input_208, v_next_83, 1.0, 0.0, 2.0, False);  input_208 = v_next_83 = None
    spike_91 = lif_forward_state_default_91[0]
    v_next_91 = lif_forward_state_default_91[1];  lif_forward_state_default_91 = None
    input_209 = torch.conv2d(spike_91, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_91 = None
    input_210 = torch.nn.functional.batch_norm(input_209, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_209 = None
    lif_forward_state_default_92 = torch.ops.snn_custom.lif_forward_state.default(input_210, v_next_84, 1.0, 0.0, 2.0, False);  input_210 = v_next_84 = None
    spike_92 = lif_forward_state_default_92[0]
    v_next_92 = lif_forward_state_default_92[1];  lif_forward_state_default_92 = None
    input_211 = torch.nn.functional.max_pool2d(spike_92, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_92 = None
    x_22 = torch.nn.functional.adaptive_avg_pool2d(input_211, (6, 6));  input_211 = None
    x_23 = x_22.flatten(1, -1);  x_22 = None
    input_212 = torch.nn.functional.dropout(x_23, 0.5, False, False);  x_23 = None
    input_213 = torch._C._nn.linear(input_212, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_212 = None
    lif_forward_state_default_93 = torch.ops.snn_custom.lif_forward_state.default(input_213, v_next_85, 1.0, 0.0, 2.0, False);  input_213 = v_next_85 = None
    spike_93 = lif_forward_state_default_93[0]
    v_next_93 = lif_forward_state_default_93[1];  lif_forward_state_default_93 = None
    input_214 = torch.nn.functional.dropout(spike_93, 0.5, False, False);  spike_93 = None
    input_215 = torch._C._nn.linear(input_214, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_214 = None
    lif_forward_state_default_94 = torch.ops.snn_custom.lif_forward_state.default(input_215, v_next_86, 1.0, 0.0, 2.0, False);  input_215 = v_next_86 = None
    spike_94 = lif_forward_state_default_94[0]
    v_next_94 = lif_forward_state_default_94[1];  lif_forward_state_default_94 = None
    input_216 = torch._C._nn.linear(spike_94, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_94 = None
    lif_forward_state_default_95 = torch.ops.snn_custom.lif_forward_state.default(input_216, v_next_87, 1.0, 0.0, 2.0, False);  input_216 = v_next_87 = None
    spike_95 = lif_forward_state_default_95[0]
    v_next_95 = lif_forward_state_default_95[1];  lif_forward_state_default_95 = None
    out_spikes_counter_11 = out_spikes_counter_10 + spike_95;  out_spikes_counter_10 = spike_95 = None
    getitem_204 = l_x_seq_[12]
    input_217 = torch.conv2d(getitem_204, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_204 = None
    input_218 = torch.nn.functional.batch_norm(input_217, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_217 = None
    lif_forward_state_default_96 = torch.ops.snn_custom.lif_forward_state.default(input_218, v_next_88, 1.0, 0.0, 2.0, False);  input_218 = v_next_88 = None
    spike_96 = lif_forward_state_default_96[0]
    v_next_96 = lif_forward_state_default_96[1];  lif_forward_state_default_96 = None
    input_219 = torch.nn.functional.max_pool2d(spike_96, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_96 = None
    input_220 = torch.conv2d(input_219, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_219 = None
    input_221 = torch.nn.functional.batch_norm(input_220, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_220 = None
    lif_forward_state_default_97 = torch.ops.snn_custom.lif_forward_state.default(input_221, v_next_89, 1.0, 0.0, 2.0, False);  input_221 = v_next_89 = None
    spike_97 = lif_forward_state_default_97[0]
    v_next_97 = lif_forward_state_default_97[1];  lif_forward_state_default_97 = None
    input_222 = torch.nn.functional.max_pool2d(spike_97, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_97 = None
    input_223 = torch.conv2d(input_222, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_222 = None
    input_224 = torch.nn.functional.batch_norm(input_223, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_223 = None
    lif_forward_state_default_98 = torch.ops.snn_custom.lif_forward_state.default(input_224, v_next_90, 1.0, 0.0, 2.0, False);  input_224 = v_next_90 = None
    spike_98 = lif_forward_state_default_98[0]
    v_next_98 = lif_forward_state_default_98[1];  lif_forward_state_default_98 = None
    input_225 = torch.conv2d(spike_98, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_98 = None
    input_226 = torch.nn.functional.batch_norm(input_225, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_225 = None
    lif_forward_state_default_99 = torch.ops.snn_custom.lif_forward_state.default(input_226, v_next_91, 1.0, 0.0, 2.0, False);  input_226 = v_next_91 = None
    spike_99 = lif_forward_state_default_99[0]
    v_next_99 = lif_forward_state_default_99[1];  lif_forward_state_default_99 = None
    input_227 = torch.conv2d(spike_99, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_99 = None
    input_228 = torch.nn.functional.batch_norm(input_227, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_227 = None
    lif_forward_state_default_100 = torch.ops.snn_custom.lif_forward_state.default(input_228, v_next_92, 1.0, 0.0, 2.0, False);  input_228 = v_next_92 = None
    spike_100 = lif_forward_state_default_100[0]
    v_next_100 = lif_forward_state_default_100[1];  lif_forward_state_default_100 = None
    input_229 = torch.nn.functional.max_pool2d(spike_100, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_100 = None
    x_24 = torch.nn.functional.adaptive_avg_pool2d(input_229, (6, 6));  input_229 = None
    x_25 = x_24.flatten(1, -1);  x_24 = None
    input_230 = torch.nn.functional.dropout(x_25, 0.5, False, False);  x_25 = None
    input_231 = torch._C._nn.linear(input_230, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_230 = None
    lif_forward_state_default_101 = torch.ops.snn_custom.lif_forward_state.default(input_231, v_next_93, 1.0, 0.0, 2.0, False);  input_231 = v_next_93 = None
    spike_101 = lif_forward_state_default_101[0]
    v_next_101 = lif_forward_state_default_101[1];  lif_forward_state_default_101 = None
    input_232 = torch.nn.functional.dropout(spike_101, 0.5, False, False);  spike_101 = None
    input_233 = torch._C._nn.linear(input_232, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_232 = None
    lif_forward_state_default_102 = torch.ops.snn_custom.lif_forward_state.default(input_233, v_next_94, 1.0, 0.0, 2.0, False);  input_233 = v_next_94 = None
    spike_102 = lif_forward_state_default_102[0]
    v_next_102 = lif_forward_state_default_102[1];  lif_forward_state_default_102 = None
    input_234 = torch._C._nn.linear(spike_102, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_102 = None
    lif_forward_state_default_103 = torch.ops.snn_custom.lif_forward_state.default(input_234, v_next_95, 1.0, 0.0, 2.0, False);  input_234 = v_next_95 = None
    spike_103 = lif_forward_state_default_103[0]
    v_next_103 = lif_forward_state_default_103[1];  lif_forward_state_default_103 = None
    out_spikes_counter_12 = out_spikes_counter_11 + spike_103;  out_spikes_counter_11 = spike_103 = None
    getitem_221 = l_x_seq_[13]
    input_235 = torch.conv2d(getitem_221, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_221 = None
    input_236 = torch.nn.functional.batch_norm(input_235, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_235 = None
    lif_forward_state_default_104 = torch.ops.snn_custom.lif_forward_state.default(input_236, v_next_96, 1.0, 0.0, 2.0, False);  input_236 = v_next_96 = None
    spike_104 = lif_forward_state_default_104[0]
    v_next_104 = lif_forward_state_default_104[1];  lif_forward_state_default_104 = None
    input_237 = torch.nn.functional.max_pool2d(spike_104, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_104 = None
    input_238 = torch.conv2d(input_237, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_237 = None
    input_239 = torch.nn.functional.batch_norm(input_238, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_238 = None
    lif_forward_state_default_105 = torch.ops.snn_custom.lif_forward_state.default(input_239, v_next_97, 1.0, 0.0, 2.0, False);  input_239 = v_next_97 = None
    spike_105 = lif_forward_state_default_105[0]
    v_next_105 = lif_forward_state_default_105[1];  lif_forward_state_default_105 = None
    input_240 = torch.nn.functional.max_pool2d(spike_105, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_105 = None
    input_241 = torch.conv2d(input_240, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_240 = None
    input_242 = torch.nn.functional.batch_norm(input_241, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_241 = None
    lif_forward_state_default_106 = torch.ops.snn_custom.lif_forward_state.default(input_242, v_next_98, 1.0, 0.0, 2.0, False);  input_242 = v_next_98 = None
    spike_106 = lif_forward_state_default_106[0]
    v_next_106 = lif_forward_state_default_106[1];  lif_forward_state_default_106 = None
    input_243 = torch.conv2d(spike_106, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_106 = None
    input_244 = torch.nn.functional.batch_norm(input_243, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_243 = None
    lif_forward_state_default_107 = torch.ops.snn_custom.lif_forward_state.default(input_244, v_next_99, 1.0, 0.0, 2.0, False);  input_244 = v_next_99 = None
    spike_107 = lif_forward_state_default_107[0]
    v_next_107 = lif_forward_state_default_107[1];  lif_forward_state_default_107 = None
    input_245 = torch.conv2d(spike_107, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_107 = None
    input_246 = torch.nn.functional.batch_norm(input_245, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_245 = None
    lif_forward_state_default_108 = torch.ops.snn_custom.lif_forward_state.default(input_246, v_next_100, 1.0, 0.0, 2.0, False);  input_246 = v_next_100 = None
    spike_108 = lif_forward_state_default_108[0]
    v_next_108 = lif_forward_state_default_108[1];  lif_forward_state_default_108 = None
    input_247 = torch.nn.functional.max_pool2d(spike_108, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_108 = None
    x_26 = torch.nn.functional.adaptive_avg_pool2d(input_247, (6, 6));  input_247 = None
    x_27 = x_26.flatten(1, -1);  x_26 = None
    input_248 = torch.nn.functional.dropout(x_27, 0.5, False, False);  x_27 = None
    input_249 = torch._C._nn.linear(input_248, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_248 = None
    lif_forward_state_default_109 = torch.ops.snn_custom.lif_forward_state.default(input_249, v_next_101, 1.0, 0.0, 2.0, False);  input_249 = v_next_101 = None
    spike_109 = lif_forward_state_default_109[0]
    v_next_109 = lif_forward_state_default_109[1];  lif_forward_state_default_109 = None
    input_250 = torch.nn.functional.dropout(spike_109, 0.5, False, False);  spike_109 = None
    input_251 = torch._C._nn.linear(input_250, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_250 = None
    lif_forward_state_default_110 = torch.ops.snn_custom.lif_forward_state.default(input_251, v_next_102, 1.0, 0.0, 2.0, False);  input_251 = v_next_102 = None
    spike_110 = lif_forward_state_default_110[0]
    v_next_110 = lif_forward_state_default_110[1];  lif_forward_state_default_110 = None
    input_252 = torch._C._nn.linear(spike_110, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_110 = None
    lif_forward_state_default_111 = torch.ops.snn_custom.lif_forward_state.default(input_252, v_next_103, 1.0, 0.0, 2.0, False);  input_252 = v_next_103 = None
    spike_111 = lif_forward_state_default_111[0]
    v_next_111 = lif_forward_state_default_111[1];  lif_forward_state_default_111 = None
    out_spikes_counter_13 = out_spikes_counter_12 + spike_111;  out_spikes_counter_12 = spike_111 = None
    getitem_238 = l_x_seq_[14]
    input_253 = torch.conv2d(getitem_238, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_238 = None
    input_254 = torch.nn.functional.batch_norm(input_253, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_253 = None
    lif_forward_state_default_112 = torch.ops.snn_custom.lif_forward_state.default(input_254, v_next_104, 1.0, 0.0, 2.0, False);  input_254 = v_next_104 = None
    spike_112 = lif_forward_state_default_112[0]
    v_next_112 = lif_forward_state_default_112[1];  lif_forward_state_default_112 = None
    input_255 = torch.nn.functional.max_pool2d(spike_112, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_112 = None
    input_256 = torch.conv2d(input_255, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_255 = None
    input_257 = torch.nn.functional.batch_norm(input_256, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_256 = None
    lif_forward_state_default_113 = torch.ops.snn_custom.lif_forward_state.default(input_257, v_next_105, 1.0, 0.0, 2.0, False);  input_257 = v_next_105 = None
    spike_113 = lif_forward_state_default_113[0]
    v_next_113 = lif_forward_state_default_113[1];  lif_forward_state_default_113 = None
    input_258 = torch.nn.functional.max_pool2d(spike_113, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_113 = None
    input_259 = torch.conv2d(input_258, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_258 = None
    input_260 = torch.nn.functional.batch_norm(input_259, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_259 = None
    lif_forward_state_default_114 = torch.ops.snn_custom.lif_forward_state.default(input_260, v_next_106, 1.0, 0.0, 2.0, False);  input_260 = v_next_106 = None
    spike_114 = lif_forward_state_default_114[0]
    v_next_114 = lif_forward_state_default_114[1];  lif_forward_state_default_114 = None
    input_261 = torch.conv2d(spike_114, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_114 = None
    input_262 = torch.nn.functional.batch_norm(input_261, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_261 = None
    lif_forward_state_default_115 = torch.ops.snn_custom.lif_forward_state.default(input_262, v_next_107, 1.0, 0.0, 2.0, False);  input_262 = v_next_107 = None
    spike_115 = lif_forward_state_default_115[0]
    v_next_115 = lif_forward_state_default_115[1];  lif_forward_state_default_115 = None
    input_263 = torch.conv2d(spike_115, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_115 = None
    input_264 = torch.nn.functional.batch_norm(input_263, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_263 = None
    lif_forward_state_default_116 = torch.ops.snn_custom.lif_forward_state.default(input_264, v_next_108, 1.0, 0.0, 2.0, False);  input_264 = v_next_108 = None
    spike_116 = lif_forward_state_default_116[0]
    v_next_116 = lif_forward_state_default_116[1];  lif_forward_state_default_116 = None
    input_265 = torch.nn.functional.max_pool2d(spike_116, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_116 = None
    x_28 = torch.nn.functional.adaptive_avg_pool2d(input_265, (6, 6));  input_265 = None
    x_29 = x_28.flatten(1, -1);  x_28 = None
    input_266 = torch.nn.functional.dropout(x_29, 0.5, False, False);  x_29 = None
    input_267 = torch._C._nn.linear(input_266, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_266 = None
    lif_forward_state_default_117 = torch.ops.snn_custom.lif_forward_state.default(input_267, v_next_109, 1.0, 0.0, 2.0, False);  input_267 = v_next_109 = None
    spike_117 = lif_forward_state_default_117[0]
    v_next_117 = lif_forward_state_default_117[1];  lif_forward_state_default_117 = None
    input_268 = torch.nn.functional.dropout(spike_117, 0.5, False, False);  spike_117 = None
    input_269 = torch._C._nn.linear(input_268, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_268 = None
    lif_forward_state_default_118 = torch.ops.snn_custom.lif_forward_state.default(input_269, v_next_110, 1.0, 0.0, 2.0, False);  input_269 = v_next_110 = None
    spike_118 = lif_forward_state_default_118[0]
    v_next_118 = lif_forward_state_default_118[1];  lif_forward_state_default_118 = None
    input_270 = torch._C._nn.linear(spike_118, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_118 = None
    lif_forward_state_default_119 = torch.ops.snn_custom.lif_forward_state.default(input_270, v_next_111, 1.0, 0.0, 2.0, False);  input_270 = v_next_111 = None
    spike_119 = lif_forward_state_default_119[0]
    v_next_119 = lif_forward_state_default_119[1];  lif_forward_state_default_119 = None
    out_spikes_counter_14 = out_spikes_counter_13 + spike_119;  out_spikes_counter_13 = spike_119 = None
    getitem_255 = l_x_seq_[15];  l_x_seq_ = None
    input_271 = torch.conv2d(getitem_255, l_self_modules_layer_modules_features_modules_0_parameters_weight_, None, (4, 4), (2, 2), (1, 1), 1);  getitem_255 = l_self_modules_layer_modules_features_modules_0_parameters_weight_ = None
    input_272 = torch.nn.functional.batch_norm(input_271, l_self_modules_layer_modules_features_modules_1_buffers_running_mean_, l_self_modules_layer_modules_features_modules_1_buffers_running_var_, l_self_modules_layer_modules_features_modules_1_parameters_weight_, l_self_modules_layer_modules_features_modules_1_parameters_bias_, False, 0.1, 1e-05);  input_271 = l_self_modules_layer_modules_features_modules_1_buffers_running_mean_ = l_self_modules_layer_modules_features_modules_1_buffers_running_var_ = l_self_modules_layer_modules_features_modules_1_parameters_weight_ = l_self_modules_layer_modules_features_modules_1_parameters_bias_ = None
    lif_forward_state_default_120 = torch.ops.snn_custom.lif_forward_state.default(input_272, v_next_112, 1.0, 0.0, 2.0, False);  input_272 = v_next_112 = None
    spike_120 = lif_forward_state_default_120[0]
    v_next_120 = lif_forward_state_default_120[1];  lif_forward_state_default_120 = None
    input_273 = torch.nn.functional.max_pool2d(spike_120, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_120 = None
    input_274 = torch.conv2d(input_273, l_self_modules_layer_modules_features_modules_4_parameters_weight_, None, (1, 1), (2, 2), (1, 1), 1);  input_273 = l_self_modules_layer_modules_features_modules_4_parameters_weight_ = None
    input_275 = torch.nn.functional.batch_norm(input_274, l_self_modules_layer_modules_features_modules_5_buffers_running_mean_, l_self_modules_layer_modules_features_modules_5_buffers_running_var_, l_self_modules_layer_modules_features_modules_5_parameters_weight_, l_self_modules_layer_modules_features_modules_5_parameters_bias_, False, 0.1, 1e-05);  input_274 = l_self_modules_layer_modules_features_modules_5_buffers_running_mean_ = l_self_modules_layer_modules_features_modules_5_buffers_running_var_ = l_self_modules_layer_modules_features_modules_5_parameters_weight_ = l_self_modules_layer_modules_features_modules_5_parameters_bias_ = None
    lif_forward_state_default_121 = torch.ops.snn_custom.lif_forward_state.default(input_275, v_next_113, 1.0, 0.0, 2.0, False);  input_275 = v_next_113 = None
    spike_121 = lif_forward_state_default_121[0]
    v_next_121 = lif_forward_state_default_121[1];  lif_forward_state_default_121 = None
    input_276 = torch.nn.functional.max_pool2d(spike_121, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_121 = None
    input_277 = torch.conv2d(input_276, l_self_modules_layer_modules_features_modules_8_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  input_276 = l_self_modules_layer_modules_features_modules_8_parameters_weight_ = None
    input_278 = torch.nn.functional.batch_norm(input_277, l_self_modules_layer_modules_features_modules_9_buffers_running_mean_, l_self_modules_layer_modules_features_modules_9_buffers_running_var_, l_self_modules_layer_modules_features_modules_9_parameters_weight_, l_self_modules_layer_modules_features_modules_9_parameters_bias_, False, 0.1, 1e-05);  input_277 = l_self_modules_layer_modules_features_modules_9_buffers_running_mean_ = l_self_modules_layer_modules_features_modules_9_buffers_running_var_ = l_self_modules_layer_modules_features_modules_9_parameters_weight_ = l_self_modules_layer_modules_features_modules_9_parameters_bias_ = None
    lif_forward_state_default_122 = torch.ops.snn_custom.lif_forward_state.default(input_278, v_next_114, 1.0, 0.0, 2.0, False);  input_278 = v_next_114 = None
    spike_122 = lif_forward_state_default_122[0]
    v_next_122 = lif_forward_state_default_122[1];  lif_forward_state_default_122 = None
    input_279 = torch.conv2d(spike_122, l_self_modules_layer_modules_features_modules_11_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_122 = l_self_modules_layer_modules_features_modules_11_parameters_weight_ = None
    input_280 = torch.nn.functional.batch_norm(input_279, l_self_modules_layer_modules_features_modules_12_buffers_running_mean_, l_self_modules_layer_modules_features_modules_12_buffers_running_var_, l_self_modules_layer_modules_features_modules_12_parameters_weight_, l_self_modules_layer_modules_features_modules_12_parameters_bias_, False, 0.1, 1e-05);  input_279 = l_self_modules_layer_modules_features_modules_12_buffers_running_mean_ = l_self_modules_layer_modules_features_modules_12_buffers_running_var_ = l_self_modules_layer_modules_features_modules_12_parameters_weight_ = l_self_modules_layer_modules_features_modules_12_parameters_bias_ = None
    lif_forward_state_default_123 = torch.ops.snn_custom.lif_forward_state.default(input_280, v_next_115, 1.0, 0.0, 2.0, False);  input_280 = v_next_115 = None
    spike_123 = lif_forward_state_default_123[0]
    v_next_123 = lif_forward_state_default_123[1];  lif_forward_state_default_123 = None
    input_281 = torch.conv2d(spike_123, l_self_modules_layer_modules_features_modules_14_parameters_weight_, None, (1, 1), (1, 1), (1, 1), 1);  spike_123 = l_self_modules_layer_modules_features_modules_14_parameters_weight_ = None
    input_282 = torch.nn.functional.batch_norm(input_281, l_self_modules_layer_modules_features_modules_15_buffers_running_mean_, l_self_modules_layer_modules_features_modules_15_buffers_running_var_, l_self_modules_layer_modules_features_modules_15_parameters_weight_, l_self_modules_layer_modules_features_modules_15_parameters_bias_, False, 0.1, 1e-05);  input_281 = l_self_modules_layer_modules_features_modules_15_buffers_running_mean_ = l_self_modules_layer_modules_features_modules_15_buffers_running_var_ = l_self_modules_layer_modules_features_modules_15_parameters_weight_ = l_self_modules_layer_modules_features_modules_15_parameters_bias_ = None
    lif_forward_state_default_124 = torch.ops.snn_custom.lif_forward_state.default(input_282, v_next_116, 1.0, 0.0, 2.0, False);  input_282 = v_next_116 = None
    spike_124 = lif_forward_state_default_124[0]
    v_next_124 = lif_forward_state_default_124[1];  lif_forward_state_default_124 = None
    input_283 = torch.nn.functional.max_pool2d(spike_124, 3, 2, 0, 1, ceil_mode = False, return_indices = False);  spike_124 = None
    x_30 = torch.nn.functional.adaptive_avg_pool2d(input_283, (6, 6));  input_283 = None
    x_31 = x_30.flatten(1, -1);  x_30 = None
    input_284 = torch.nn.functional.dropout(x_31, 0.5, False, False);  x_31 = None
    input_285 = torch._C._nn.linear(input_284, l_self_modules_layer_modules_classifier_modules_1_parameters_weight_, None);  input_284 = l_self_modules_layer_modules_classifier_modules_1_parameters_weight_ = None
    lif_forward_state_default_125 = torch.ops.snn_custom.lif_forward_state.default(input_285, v_next_117, 1.0, 0.0, 2.0, False);  input_285 = v_next_117 = None
    spike_125 = lif_forward_state_default_125[0]
    v_next_125 = lif_forward_state_default_125[1];  lif_forward_state_default_125 = None
    input_286 = torch.nn.functional.dropout(spike_125, 0.5, False, False);  spike_125 = None
    input_287 = torch._C._nn.linear(input_286, l_self_modules_layer_modules_classifier_modules_4_parameters_weight_, None);  input_286 = l_self_modules_layer_modules_classifier_modules_4_parameters_weight_ = None
    lif_forward_state_default_126 = torch.ops.snn_custom.lif_forward_state.default(input_287, v_next_118, 1.0, 0.0, 2.0, False);  input_287 = v_next_118 = None
    spike_126 = lif_forward_state_default_126[0]
    v_next_126 = lif_forward_state_default_126[1];  lif_forward_state_default_126 = None
    input_288 = torch._C._nn.linear(spike_126, l_self_modules_layer_modules_classifier_modules_6_parameters_weight_, None);  spike_126 = l_self_modules_layer_modules_classifier_modules_6_parameters_weight_ = None
    lif_forward_state_default_127 = torch.ops.snn_custom.lif_forward_state.default(input_288, v_next_119, 1.0, 0.0, 2.0, False);  input_288 = v_next_119 = None
    spike_127 = lif_forward_state_default_127[0]
    v_next_127 = lif_forward_state_default_127[1];  lif_forward_state_default_127 = None
    out_spikes_counter_15 = out_spikes_counter_14 + spike_127;  out_spikes_counter_14 = spike_127 = None
    truediv = out_spikes_counter_15 / 16;  out_spikes_counter_15 = None
    return (truediv, v_next_120, v_next_121, v_next_122, v_next_123, v_next_124, v_next_125, v_next_126, v_next_127)
    