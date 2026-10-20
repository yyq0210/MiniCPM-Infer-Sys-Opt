# 固定安装源码的行号摘录

采集日期：2026-10-20。以下 vLLM 代码版权归 vLLM contributors，SPDX-License-Identifier: Apache-2.0。保留文件哈希以区别厂商镜像与网上最新分支。完整相对路径以 `/usr/local/lib/python3.12/site-packages/` 为根。摘录左列为原文件行号；此文件只用作阅读证据。

[所有已读取文件的 SHA256](source_index.json)。

## vllm/model_executor/models/llama.py

SHA256: `1e1003454985bf86d3868a7755bcf5324676881548b3895809edf0b76698d384`

原文件 118–122 行：

```text
  118     def forward(self, x):
  119         x, _ = self.gate_up_proj(x)
  120         x = self.act_fn(x)
  121         x, _ = self.down_proj(x)
  122         return x
```

原文件 224–234 行：

```text
  224     def forward(
  225         self,
  226         positions: torch.Tensor,
  227         hidden_states: torch.Tensor,
  228     ) -> torch.Tensor:
  229         qkv, _ = self.qkv_proj(hidden_states)
  230         q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
  231         q, k = self.rotary_emb(positions, q, k)
  232         attn_output = self.attn(q, k, v)
  233         output, _ = self.o_proj(attn_output)
  234         return output
```

原文件 313–344 行：

```text
  313     def forward(
  314         self,
  315         positions: torch.Tensor,
  316         hidden_states: torch.Tensor,
  317         residual: torch.Tensor | None,
  318     ) -> tuple[torch.Tensor, torch.Tensor]:
  319         # Self Attention
  320         if residual is None:
  321             residual = hidden_states
  322             hidden_states = self.input_layernorm(hidden_states)
  323         else:
  324             hidden_states, residual = self.input_layernorm(hidden_states, residual)
  325         hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
  326 
  327         # Fully Connected
  328         hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
  329         hidden_states = self.mlp(hidden_states)
  330         return hidden_states, residual
  331 
  332     def get_quant_config(self, vllm_config: VllmConfig) -> QuantizationConfig | None:
  333         """Get quantization config for this layer. Override in subclasses."""
  334         return vllm_config.quant_config
  335 
  336 
  337 @support_torch_compile(
  338     # TODO[#32068]: Investigate recompilation
  339     # mark_unbacked_dims={"input_ids": 0},
  340     dynamic_arg_dims={
  341         "input_ids": {0: "b"},
  342         "positions": {0: "b"},
  343         "intermediate_tensors": {0: "b"},
  344         "inputs_embeds": {0: "b"},
```

原文件 392–431 行：

```text
  392     def forward(
  393         self,
  394         input_ids: torch.Tensor | None,
  395         positions: torch.Tensor,
  396         intermediate_tensors: IntermediateTensors | None,
  397         inputs_embeds: torch.Tensor | None = None,
  398         **extra_layer_kwargs,
  399     ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
  400         if get_pp_group().is_first_rank:
  401             if inputs_embeds is not None:
  402                 hidden_states = inputs_embeds
  403             else:
  404                 hidden_states = self.embed_input_ids(input_ids)
  405             residual = None
  406         else:
  407             assert intermediate_tensors is not None
  408             hidden_states = intermediate_tensors["hidden_states"]
  409             residual = intermediate_tensors["residual"]
  410 
  411         aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
  412         for idx, layer in enumerate(
  413             islice(self.layers, self.start_layer, self.end_layer)
  414         ):
  415             hidden_states, residual = layer(
  416                 positions, hidden_states, residual, **extra_layer_kwargs
  417             )
  418             self._maybe_add_hidden_state(
  419                 aux_hidden_states, idx + 1, hidden_states, residual
  420             )
  421 
  422         if not get_pp_group().is_last_rank:
  423             return IntermediateTensors(
  424                 {"hidden_states": hidden_states, "residual": residual}
  425             )
  426 
  427         hidden_states, _ = self.norm(hidden_states, residual)
  428 
  429         if len(aux_hidden_states) > 0:
  430             return hidden_states, aux_hidden_states
  431         return hidden_states
```

## vllm/model_executor/layers/layernorm.py

SHA256: `696615d858292fea7a2942480803cd4c7d2d1e88d818bfe5cd9d695ac464f37c`

原文件 74–94 行：

```text
   74     def forward_native(
   75         self,
   76         x: torch.Tensor,
   77         residual: torch.Tensor | None = None,
   78     ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
   79         """PyTorch-native implementation equivalent to forward()."""
   80         if residual is None:
   81             return ir.ops.rms_norm(
   82                 x,
   83                 self.weight.data if self.pass_weight else None,
   84                 self.variance_epsilon,
   85                 self.variance_size_override,
   86             )
   87         else:
   88             return ir.ops.fused_add_rms_norm.maybe_inplace(
   89                 x,
   90                 residual,
   91                 self.weight.data if self.pass_weight_add else None,
   92                 self.variance_epsilon,
   93                 self.variance_size_override,
   94             )
```

## vllm/ir/ops/layernorm.py

SHA256: `d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910`

原文件 1–64 行：

```text
    1 # SPDX-License-Identifier: Apache-2.0
    2 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    3 import torch
    4 from torch import Tensor
    5 
    6 from ..op import register_op
    7 
    8 
    9 @register_op
   10 def rms_norm(
   11     x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
   12 ) -> Tensor:
   13     """Weighted root-mean-square layer normalization"""
   14     orig_dtype = x.dtype
   15     x = x.to(torch.float32)
   16     x_var = x if variance_size is None else x[..., :variance_size]
   17     variance = x_var.pow(2).mean(dim=-1, keepdim=True)
   18     x = x * torch.rsqrt(variance + epsilon)
   19     if weight is not None:
   20         x = x.to(weight.dtype) * weight
   21     return x.to(orig_dtype)
   22 
   23 
   24 @rms_norm.register_input_generator
   25 def _rms_norm_input_generator(
   26     num_tokens: int, hidden_size: int, dtype: torch.dtype, epsilon: float = 1e-5
   27 ) -> tuple:
   28     x = torch.randn(num_tokens, hidden_size, dtype=dtype)
   29     weight = torch.randn(hidden_size, dtype=dtype)
   30     return x, weight, epsilon
   31 
   32 
   33 # Reductions in rms_norm accumulate rounding error at large shapes
   34 # (e.g. 32768x16384), causing a few elements out of millions to exceed
   35 # the default float16 tolerance.
   36 rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)
   37 
   38 
   39 @register_op(allow_inplace=True)
   40 def fused_add_rms_norm(
   41     x: Tensor,
   42     x_residual: Tensor,
   43     weight: Tensor | None,
   44     epsilon: float,
   45     variance_size: int | None = None,
   46 ) -> tuple[Tensor, Tensor]:
   47     """Fused add and weighted root-mean-square layer normalization"""
   48     orig_dtype = x.dtype
   49     x = x.to(torch.float32)
   50     x = x + x_residual.to(torch.float32)
   51     x_residual = x.to(orig_dtype)
   52 
   53     x_var = x if variance_size is None else x[..., :variance_size]
   54     variance = x_var.pow(2).mean(dim=-1, keepdim=True)
   55     x = x * torch.rsqrt(variance + epsilon)
   56     if weight is not None:
   57         x = x.to(weight.dtype) * weight
   58     return x.to(orig_dtype), x_residual
   59 
   60 
   61 # fused_add_rms_norm has similar rounding error accumulation as rms_norm
   62 fused_add_rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)
   63 
   64 
```

## vllm/model_executor/layers/linear.py

SHA256: `0bd60103d85d7183791077c46e0770d0a2adc8e8811b162b9084c3cbe92ba662`

原文件 217–226 行：

```text
  217     def apply(
  218         self,
  219         layer: torch.nn.Module,
  220         x: torch.Tensor,
  221         bias: torch.Tensor | None = None,
  222     ) -> torch.Tensor:
  223         if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
  224             return linear_batch_invariant(x, layer.weight, bias)
  225         return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
  226 
```

## vllm/model_executor/layers/utils.py

SHA256: `061adffc6af55ef3b001557c0fad738e474c5e4134fa3dc35cb967bdbc90a121`

原文件 92–98 行：

```text
   92 def default_unquantized_gemm(
   93     layer: torch.nn.Module,
   94     x: torch.Tensor,
   95     weight: torch.Tensor,
   96     bias: torch.Tensor | None = None,
   97 ):
   98     return torch.nn.functional.linear(x, weight, bias)
```

原文件 332–338 行：

```text
  332 def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
  333     if current_platform.is_rocm():
  334         return rocm_unquantized_gemm
  335     elif current_platform.is_cpu():
  336         return cpu_unquantized_gemm
  337     else:
  338         return default_unquantized_gemm
```

## vllm/model_executor/layers/activation.py

SHA256: `757209cef4b5cc091d29d95b4a6b29f2511984110c9b66810c760d4499cb156c`

原文件 118–144 行：

```text
  118 class SiluAndMul(CustomOp):
  119     """An activation function for SwiGLU.
  120 
  121     The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.
  122 
  123     Shapes:
  124         x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
  125         return: (num_tokens, d) or (batch_size, seq_len, d)
  126     """
  127 
  128     # --8<-- [end:silu_and_mul]
  129 
  130     def __init__(self, *, compile_native: bool = True):
  131         super().__init__(compile_native=compile_native)
  132         if current_platform.is_cuda_alike() or current_platform.is_xpu():
  133             self.op = torch.ops._C.silu_and_mul
  134         elif current_platform.is_cpu():
  135             self._forward_method = self.forward_native
  136 
  137     @staticmethod
  138     def forward_native(x: torch.Tensor) -> torch.Tensor:
  139         """PyTorch-native implementation equivalent to forward()."""
  140         d = x.shape[-1] // 2
  141         return F.silu(x[..., :d]) * x[..., d:]
  142 
  143     def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
  144         d = x.shape[-1] // 2
```

## vllm/v1/attention/backends/triton_attn.py

SHA256: `1219c4d1820b56e99ef6686de8a49c8c5aa57707794bcabf3322023799a03605`

原文件 52–56 行：

```text
   52 
   53 # constants
   54 MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
   55 NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments
   56 
```

原文件 136–170 行：

```text
  136 
  137         # Modify the threshold if needed.
  138         if self.decode_cudagraph_enabled:
  139             capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
  140             assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."
  141 
  142             # Select the CUDA Graph capture size closest to self.seq_threshold_3D
  143             # as threshold. This ensures that each captured graph covers the
  144             # correct execution path.
  145             self.seq_threshold_3D = min(
  146                 capture_sizes,
  147                 key=lambda x: abs(x - self.seq_threshold_3D),
  148             )
  149 
  150         self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
  151         headdim_padded = next_power_of_2(self.headdim)
  152         self.softmax_segm_output = torch.empty(
  153             (
  154                 self.seq_threshold_3D,
  155                 self.num_heads_q,
  156                 self.num_par_softmax_segments,
  157                 headdim_padded,
  158             ),
  159             dtype=torch.float32,
  160             device=device,
  161         )
  162         self.softmax_segm_max = torch.empty(
  163             (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
  164             dtype=torch.float32,
  165             device=device,
  166         )
  167         self.softmax_segm_expsum = torch.empty(
  168             (self.seq_threshold_3D, self.num_heads_q, self.num_par_softmax_segments),
  169             dtype=torch.float32,
  170             device=device,
```

原文件 650–690 行：

```text
  650             max_seqlen_k=max_seqlen_k,
  651             softmax_scale=self.scale,
  652             causal=attn_metadata.causal,
  653             alibi_slopes=self.alibi_slopes,
  654             use_alibi_sqrt=self.use_alibi_sqrt,
  655             window_size=self.sliding_window,
  656             block_table=block_table,
  657             softcap=self.logits_soft_cap,
  658             q_descale=q_descale,
  659             k_descale=k_descale,
  660             v_descale=v_descale,
  661             seq_threshold_3D=seq_threshold_3D,
  662             num_par_softmax_segments=num_par_softmax_segments,
  663             softmax_segm_output=softmax_segm_output,
  664             softmax_segm_max=softmax_segm_max,
  665             softmax_segm_expsum=softmax_segm_expsum,
  666             sinks=self.sinks,
  667             output_scale=output_scale,
  668             mm_prefix_range=mm_prefix_range_tensor,
  669             kv_quant_mode=self._kv_quant_mode,
  670             k_scale_cache=k_scale_cache,
  671             v_scale_cache=v_scale_cache,
  672             chunk_lookback=self.chunk_lookback,
  673             use_td=self.use_td,
  674         )
  675 
  676         return output
  677 
  678     def _forward_encoder_attention(
  679         self,
  680         query: torch.Tensor,
  681         key: torch.Tensor,
  682         value: torch.Tensor,
  683         output: torch.Tensor,
  684         attn_metadata: TritonAttentionMetadata,
  685         layer: torch.nn.Module,
  686     ) -> torch.Tensor:
  687         """Forward pass for encoder attention without KV cache.
  688 
  689         Args:
  690             query: shape = [num_encoder_tokens, num_heads, head_size]
```

原文件 724–808 行：

```text
  724     def do_kv_cache_update(
  725         self,
  726         layer: AttentionLayer,
  727         key: torch.Tensor,
  728         value: torch.Tensor,
  729         kv_cache: torch.Tensor,
  730         slot_mapping: torch.Tensor,
  731     ):
  732         if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
  733             # For encoder attention,
  734             # we use direct Q, K, V tensors without caching
  735             return
  736         # Reshape the input keys and values and store them in the cache.
  737         if self._is_per_token_head_quant:
  738             self._ensure_scale_caches(kv_cache)
  739             key_cache, value_cache = kv_cache.unbind(1)
  740             if key_cache.dtype == torch.uint8:
  741                 key_cache = key_cache.view(self.fp8_dtype)
  742                 value_cache = value_cache.view(self.fp8_dtype)
  743             triton_reshape_and_cache_flash_per_token_head_quant(
  744                 key,
  745                 value,
  746                 key_cache,
  747                 value_cache,
  748                 self._k_scale_cache,
  749                 self._v_scale_cache,
  750                 slot_mapping,
  751             )
  752             return
  753         # For decoder and cross-attention, use KV cache as before.
  754         key_cache, value_cache = kv_cache.unbind(1)
  755         if is_quantized_kv_cache(self.kv_cache_dtype):
  756             key_cache = key_cache.view(self.fp8_dtype)
  757             value_cache = value_cache.view(self.fp8_dtype)
  758         triton_reshape_and_cache_flash(
  759             key,
  760             value,
  761             key_cache,
  762             value_cache,
  763             slot_mapping,
  764             self.kv_cache_dtype,
  765             layer._k_scale,
  766             layer._v_scale,
  767         )
  768 
  769     def fused_rope_kvcache_supported(self):
  770         if self._is_per_token_head_quant:
  771             return False
  772         return rocm_aiter_ops.is_enabled()
  773 
  774     def do_rope_and_kv_cache_update(
  775         self,
  776         layer: AttentionLayer,
  777         query: torch.Tensor,
  778         key: torch.Tensor,
  779         value: torch.Tensor,
  780         positions: torch.Tensor,
  781         cos_sin_cache: torch.Tensor,
  782         is_neox: bool,
  783         kv_cache: torch.Tensor,
  784         layer_slot_mapping: torch.Tensor,
  785     ):
  786         key_cache, value_cache = kv_cache.unbind(1)
  787         flash_layout = True
  788 
  789         is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
  790         if is_fp8_kv_cache:
  791             key_cache = key_cache.view(self.fp8_dtype)
  792             value_cache = value_cache.view(self.fp8_dtype)
  793 
  794         rocm_aiter_ops.triton_rope_and_cache(
  795             query,
  796             key,
  797             value,
  798             positions,
  799             cos_sin_cache,
  800             is_neox,
  801             key_cache,
  802             value_cache,
  803             layer_slot_mapping,
  804             layer._k_scale,
  805             layer._v_scale,
  806             flash_layout,
  807             is_fp8_kv_cache,
  808         )
```

## vllm/v1/attention/ops/triton_unified_attention.py

SHA256: `bab2c0b9f9a7793ca411796b48ebb4734433160e7e99eabf46ddb7f1c30cb6e0`

原文件 288–328 行：

```text
  288     kv_head_idx = tl.program_id(1)
  289     segm_idx = tl.program_id(2) if IS_3D else 0
  290 
  291     (
  292         seq_idx,
  293         q_block_local_idx,
  294         cur_batch_in_all_start_index,
  295         cur_batch_query_len,
  296         seq_len,
  297     ) = resolve_seq_and_query_len(
  298         query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
  299     )
  300 
  301     if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
  302         return
  303 
  304     if IS_3D:
  305         tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
  306         if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
  307             return
  308     else:
  309         tiles_per_segment = 0
  310 
  311     # Number of valid query rows in this block (used by TD descriptor
  312     # shapes, but always computed so the variable stays in scope).
  313     q_block_local_len = tl.minimum(
  314         BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q
  315     )
  316 
  317     offs_m = tl.arange(0, BLOCK_M)
  318     offs_d = tl.arange(0, HEAD_SIZE_PADDED)
  319     offs_t = tl.arange(0, TILE_SIZE)
  320     query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
  321 
  322     query_offset_0 = cur_batch_in_all_start_index + query_pos
  323     query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
  324     query_offset = (
  325         query_offset_0[:, None] * query_stride_0
  326         + query_offset_1[:, None] * query_stride_1
  327         + offs_d[None, :]
  328     )
```

原文件 400–411 行：

```text
  400 
  401     # iterate through tiles (now limited to the sliding window range)
  402     for j in range(loop_lo, loop_hi):
  403         seq_offset = j * TILE_SIZE + offs_t
  404         tile_mask = seq_offset < max_seq_prefix_len
  405 
  406         physical_block_idx = tl.load(
  407             block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
  408         ).to(tl.int64)
  409 
  410         if USE_TD:
  411             # All TILE_SIZE slots within a single KV tile map to one
```

原文件 449–465 行：

```text
  449         else:
  450             v_offset = (
  451                 physical_block_idx[:, None] * stride_v_cache_0
  452                 + kv_head_idx * stride_v_cache_2
  453                 + offs_d[None, :] * stride_v_cache_3
  454                 + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
  455             )
  456             k_offset = (
  457                 physical_block_idx[None, :] * stride_k_cache_0
  458                 + kv_head_idx * stride_k_cache_2
  459                 + offs_d[:, None] * stride_k_cache_3
  460                 + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
  461             )
  462             # K : (HEAD_SIZE, TILE_SIZE)
  463             K_load = tl.load(
  464                 key_cache_ptr + k_offset,
  465                 mask=dim_mask[:, None] & tile_mask[None, :],
```

原文件 513–566 行：

```text
  513         # S : (BLOCK_M, TILE_SIZE)
  514         S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
  515         if USE_PER_TOKEN_HEAD_SCALES:
  516             # Per-token-head quant: fuse softmax_scale with per-head k_scale
  517             # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
  518             S += tl.dot(Q, K) * (score_scale * k_token_head_scales[None, :])
  519         else:
  520             S += score_scale * tl.dot(Q, K)
  521 
  522         if USE_SOFTCAP:
  523             S = apply_softcap(S, softcap)
  524 
  525         S = tl.where(
  526             query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
  527         )
  528 
  529         if USE_ALIBI_SLOPES:
  530             S = apply_alibi_to_score(
  531                 S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
  532             )
  533 
  534         if USE_QQ_BIAS:
  535             S += load_qq_bias_tile(
  536                 qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
  537             )
  538 
  539         M, L, P, alpha = softmax_step(S, M, L)
  540         acc = acc * alpha[:, None]
  541 
  542         if SLIDING_WINDOW:
  543             qpos_lo = q_block_local_idx * BLOCK_Q
  544             dist = context_len + qpos_lo - seq_offset[:, None]
  545             if USE_PER_SEQ_CAUSAL:
  546                 is_causal_seq = tl.load(per_seq_causal_ptr + seq_idx)
  547                 sw_mask_v = tl.where(
  548                     is_causal_seq,
  549                     dist < SLIDING_WINDOW,
  550                     (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW),
  551                 )
  552             elif USE_CAUSAL:
  553                 sw_mask_v = dist < SLIDING_WINDOW
  554             else:
  555                 sw_mask_v = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)
  556             V = tl.where(sw_mask_v, V, 0.0)
  557         if USE_PER_TOKEN_HEAD_SCALES:
  558             # Per-token-head quant: apply v_scale to P instead of V.
  559             P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
  560             acc += tl.dot(P_v, V)
  561         else:
  562             acc += tl.dot(P.to(V.dtype), V)
  563 
  564     # ---- Epilogue ---------------------------------------------------------
  565     if IS_3D:
  566         if USE_FP8_Q_DESCALE:
```

原文件 856–865 行：

```text
  856     num_kv_heads = k.shape[2]
  857     num_queries_per_kv = num_query_heads // num_kv_heads
  858     head_size = q.shape[2]
  859 
  860     BLOCK_M = (
  861         16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
  862     )
  863     BLOCK_Q = BLOCK_M // num_queries_per_kv
  864 
  865     # Tuned launch parameters; ``None`` lets Triton pick its defaults.
```

原文件 969–978 行：

```text
  969     use_3d = not (
  970         seq_threshold_3D is None
  971         or num_par_softmax_segments is None
  972         or softmax_segm_output is None
  973         or softmax_segm_max is None
  974         or softmax_segm_expsum is None
  975         or max_seqlen_q > 1
  976         or num_seqs > seq_threshold_3D
  977         or is_batch_invariant
  978     )
```

原文件 1003–1012 行：

```text
 1003     num_segments = num_par_softmax_segments if use_3d else 1
 1004 
 1005     grid: tuple[Any, ...]
 1006     if not use_3d:
 1007         grid = (total_num_q_blocks, num_kv_heads)
 1008         tile_size = TILE_SIZE_PREFILL
 1009     else:
 1010         grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
 1011         tile_size = TILE_SIZE_DECODE
 1012 
```

原文件 1093–1114 行：

```text
 1093 
 1094     if use_3d:
 1095         reduce_segments[(q.shape[0], num_query_heads)](
 1096             output_ptr=out,
 1097             segm_output_ptr=softmax_segm_output,
 1098             segm_max_ptr=softmax_segm_max,
 1099             segm_expsum_ptr=softmax_segm_expsum,
 1100             seq_lens_ptr=seqused_k,
 1101             num_seqs=num_seqs,
 1102             num_query_heads=num_query_heads,
 1103             out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
 1104             output_stride_0=out.stride(0),
 1105             output_stride_1=out.stride(1),
 1106             block_table_stride=block_table.stride(0),
 1107             TILE_SIZE=TILE_SIZE_DECODE,
 1108             HEAD_SIZE=head_size,
 1109             HEAD_SIZE_PADDED=head_size_padded,
 1110             query_start_len_ptr=cu_seqlens_q,
 1111             BLOCK_Q=BLOCK_Q,
 1112             NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
 1113             USE_FP8=output_scale is not None,
 1114         )
```

## vllm/v1/attention/ops/triton_reshape_and_cache_flash.py

SHA256: `f18224e51162e37c14a2e2b3045bca55baf0b207b1d203ad6ee7d8ed417139eb`

原文件 33–70 行：

```text
   33 def reshape_and_cache_kernel_flash(
   34     key_ptr,  # [num_tokens, num_heads, head_size]
   35     value_ptr,  # [num_tokens, num_heads, head_size]
   36     key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
   37     value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
   38     slot_mapping_ptr,  # [num_tokens]
   39     k_scale,  # float32
   40     v_scale,  # float32
   41     # strides
   42     key_stride: tl.int64,
   43     value_stride: tl.int64,
   44     block_stride: tl.int64,
   45     head_stride: tl.int64,
   46     dim_stride_k: tl.int64,
   47     dim_stride_v: tl.int64,
   48     page_stride: tl.int64,
   49     num_heads: tl.constexpr,
   50     head_size: tl.constexpr,
   51     block_size: tl.constexpr,
   52     x: tl.constexpr,
   53     USE_HEAD_MAJOR_LAYOUT: tl.constexpr,
   54     # FP8 flags
   55     FP8_KV_CACHE: tl.constexpr,
   56     # tune parameters
   57     TILE_SIZE: tl.constexpr,
   58 ):
   59     token_idx = tl.program_id(axis=0)
   60     slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
   61     if slot_idx < 0:
   62         # Padding token that should be ignored.
   63         return
   64 
   65     block_idx = slot_idx // block_size
   66     block_offset = slot_idx % block_size
   67 
   68     tile_i = tl.program_id(axis=1)
   69     tile_offs = tl.arange(0, TILE_SIZE)
   70     tile_pos = tile_i * TILE_SIZE + tile_offs
```

原文件 344–363 行：

```text
  344 def triton_reshape_and_cache_flash(
  345     key: torch.Tensor,  # [num_tokens, num_heads, head_size]
  346     value: torch.Tensor,  # [num_tokens, num_heads, head_size]
  347     # [num_blocks, block_size, num_heads, head_size]
  348     key_cache: torch.Tensor,
  349     # [num_blocks, block_size, num_heads, head_size]
  350     value_cache: torch.Tensor,
  351     slot_mapping: torch.Tensor,  # [num_tokens]
  352     kv_cache_dtype: str,  # "auto", "fp8"
  353     k_scale: torch.Tensor,  # float32
  354     v_scale: torch.Tensor,  # float32
  355 ):
  356     num_heads = key.shape[1]
  357     head_size = key.shape[2]
  358 
  359     use_head_major_layout = key_cache.ndim == 5
  360     if use_head_major_layout:
  361         block_size = key_cache.shape[3]
  362         x = key_cache.shape[4]
  363         head_stride = key_cache.stride(1)
```

## vllm/compilation/passes/fusion/rope_kvcache_fusion.py

SHA256: `faed3153db71f91bddcccf99cfab47dca7881933769d262dec52e22b5bfad163`

原文件 46–98 行：

```text
   46 def fused_rope_and_unified_kv_cache_update_impl(
   47     query: torch.Tensor,
   48     key: torch.Tensor,
   49     value: torch.Tensor,
   50     positions: torch.Tensor,
   51     cos_sin_cache: torch.Tensor,
   52     is_neox: bool,
   53     layer_name: LayerNameType,
   54 ) -> torch.Tensor:
   55     """
   56     This impl fetches the KV cache and slot mapping from the forward context,
   57     then calls the layer impl's `AttentionImpl.do_rope_and_kv_cache_update` method.
   58     It also returns a dummy tensor, similar to `Attention.unified_kv_cache_update`,
   59     that is passed to unified_attention to signal a side effect and
   60     the data dependency between them to ensure torch.compile preserves ordering.
   61     """
   62     layer_name = _resolve_layer_name(layer_name)
   63     _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
   64     if layer_slot_mapping is not None:
   65         attn_layer.impl.do_rope_and_kv_cache_update(
   66             attn_layer,
   67             query,
   68             key,
   69             value,
   70             positions,
   71             cos_sin_cache,
   72             is_neox,
   73             kv_cache,
   74             layer_slot_mapping,
   75         )
   76 
   77     return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
   78 
   79 
   80 def fused_rope_and_unified_kv_cache_update_fake(
   81     query: torch.Tensor,
   82     key: torch.Tensor,
   83     value: torch.Tensor,
   84     positions: torch.Tensor,
   85     cos_sin_cache: torch.Tensor,
   86     is_neox: bool,
   87     layer_name: LayerNameType,
   88 ) -> torch.Tensor:
   89     return torch.empty(0, device=query.device, dtype=query.dtype)
   90 
   91 
   92 direct_register_custom_op(
   93     op_name="fused_rope_and_unified_kv_cache_update",
   94     op_func=fused_rope_and_unified_kv_cache_update_impl,
   95     mutates_args=["query", "key"],
   96     fake_impl=fused_rope_and_unified_kv_cache_update_fake,
   97 )
   98 
```

原文件 328–356 行：

```text
  328     def _mk_pattern_with_layer_name_input(self, _ln):
  329         """Pattern/replacement with layer_name as an explicit input."""
  330 
  331         def pattern(qkv, positions, cos_sin_cache, layer_name):
  332             q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
  333             q, k = self.rope_matcher(positions, q, k, cos_sin_cache)
  334             q = q.view(-1, self.num_heads, self.head_size)
  335             k = k.view(-1, self.num_kv_heads, self.head_size)
  336             v = v.view(-1, self.num_kv_heads, self.head_size_v)
  337             return torch.ops.vllm.unified_kv_cache_update(k, v, layer_name), q, k, v
  338 
  339         def replacement(qkv, positions, cos_sin_cache, layer_name):
  340             q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
  341             q = q.view(-1, self.num_heads, self.head_size)
  342             k = k.view(-1, self.num_kv_heads, self.head_size)
  343             v = v.view(-1, self.num_kv_heads, self.head_size_v)
  344             results = auto_functionalized(
  345                 self.FUSED_OP,
  346                 query=q,
  347                 key=k,
  348                 value=v,
  349                 positions=positions,
  350                 cos_sin_cache=cos_sin_cache,
  351                 is_neox=self.is_neox,
  352                 layer_name=layer_name,
  353             )
  354             return results[0], results[1], results[2], v
  355 
  356         return pattern, replacement
```

原文件 426–455 行：

```text
  426     def __init__(self, config: VllmConfig) -> None:
  427         super().__init__(config)
  428 
  429         self.patterns: PatternMatcherPass = PatternMatcherPass(
  430             pass_name="rope_kv_cache_fusion_pass"
  431         )
  432 
  433         cc = config.compilation_config
  434         self.max_token_num = cc.pass_config.rope_kvcache_fusion_max_token_num
  435 
  436         attn_layers = get_layers_from_vllm_config(config, Attention)
  437         # When _USE_LAYERNAME is enabled, layer_name is a wildcard so all
  438         # layers produce the same pattern — register once then break.
  439         for _, layer in attn_layers.items():
  440             if layer.impl.fused_rope_kvcache_supported():
  441                 for is_neox in [True, False]:
  442                     if _supports_static_q_fp8_quant_fusion():
  443                         RopeStaticQQuantKVCachePattern(
  444                             layer=layer,
  445                             is_neox=is_neox,
  446                         ).register(self.patterns)
  447                     RopeReshapeKVCachePattern(
  448                         layer=layer,
  449                         is_neox=is_neox,
  450                     ).register(self.patterns)
  451                 if _USE_LAYERNAME:
  452                     break
  453 
  454         self.dump_patterns(config, self.patterns)
  455 
```

## vllm/v1/core/sched/scheduler.py

SHA256: `df78076164fe02a63876d7690cefac810f1ba4d8fb58afc15e786c1608064090`

原文件 388–416 行：

```text
  388     def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
  389         self.current_step += 1
  390         # NOTE(woosuk) on the scheduling algorithm:
  391         # There's no "decoding phase" nor "prefill phase" in the scheduler.
  392         # Each request just has the num_computed_tokens and
  393         # num_tokens_with_spec. num_tokens_with_spec =
  394         # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
  395         # At each step, the scheduler tries to assign tokens to the requests
  396         # so that each request's num_computed_tokens can catch up its
  397         # num_tokens_with_spec. This is general enough to cover
  398         # chunked prefills, prefix caching, speculative decoding,
  399         # and the "jump decoding" optimization in the future.
  400 
  401         scheduled_new_reqs: list[Request] = []
  402         scheduled_resumed_reqs: list[Request] = []
  403         scheduled_running_reqs: list[Request] = []
  404         preempted_reqs: list[Request] = []
  405 
  406         req_to_new_blocks: dict[str, KVCacheBlocks] = {}
  407         num_scheduled_tokens: dict[str, int] = {}
  408         token_budget = self.max_num_scheduled_tokens
  409         if self._pause_state == PauseState.PAUSED_ALL:
  410             # Do not schedule any requests when paused.
  411             token_budget = 0
  412 
  413         # Encoder-related.
  414         scheduled_encoder_inputs: dict[str, list[int]] = {}
  415         encoder_compute_budget = self.max_num_encoder_input_tokens
  416         # Spec decode-related.
```

原文件 523–538 行：

```text
  523             with record_function_or_nullcontext("schedule: allocate_slots"):
  524                 while True:
  525                     new_blocks = self.kv_cache_manager.allocate_slots(
  526                         request,
  527                         num_new_tokens,
  528                         num_lookahead_tokens=self.num_lookahead_tokens,
  529                     )
  530 
  531                     if new_blocks is not None:
  532                         # The request can be scheduled.
  533                         break
  534 
  535                     # The request cannot be scheduled.
  536                     # Preempt the lowest-priority request.
  537                     if self.policy == SchedulingPolicy.PRIORITY:
  538                         preempted_req = max(
```

原文件 643–655 行：

```text
  643                     if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
  644                         logger.debug(
  645                             "%s is still in WAITING_FOR_REMOTE_KVS state.",
  646                             request_id,
  647                         )
  648                     request_queue.pop_request()
  649                     step_skipped_waiting.prepend_request(request)
  650                     continue
  651 
  652                 # Check that adding the request still respects the max_loras
  653                 # constraint.
  654                 if (
  655                     self.lora_config
```

原文件 874–900 行：

```text
  874                 new_blocks = self.kv_cache_manager.allocate_slots(
  875                     request,
  876                     num_new_tokens,
  877                     num_new_computed_tokens=num_new_local_computed_tokens,
  878                     new_computed_blocks=new_computed_blocks,
  879                     num_lookahead_tokens=effective_lookahead_tokens,
  880                     num_external_computed_tokens=num_external_computed_tokens,
  881                     delay_cache_blocks=load_kv_async,
  882                     num_encoder_tokens=num_encoder_tokens,
  883                     full_sequence_must_fit=self.scheduler_reserve_full_isl,
  884                     reserved_blocks=reserved_blocks,
  885                     has_scheduled_reqs=bool(self.running),
  886                 )
  887 
  888                 if new_blocks is None:
  889                     # The request cannot be scheduled.
  890 
  891                     # NOTE: we need to untouch the request from the encode cache
  892                     # manager
  893                     if request.has_encoder_inputs:
  894                         self.encoder_cache_manager.free(request)
  895                     break
  896 
  897                 # KVTransfer: the connector uses this info to determine
  898                 # if a load is needed. Note that
  899                 # This information is used to determine if a load is
  900                 # needed for this request.
```

原文件 1107–1132 行：

```text
 1107     def _preempt_request(self, request: Request, timestamp: float) -> None:
 1108         """Preempt a request and put it back to the waiting queue.
 1109 
 1110         NOTE: The request should be popped from the running queue outside of this
 1111         method.
 1112         """
 1113         assert request.status == RequestStatus.RUNNING, (
 1114             "Only running requests can be preempted"
 1115         )
 1116         self._free_request_blocks(request)
 1117         self.encoder_cache_manager.free(request)
 1118         self._inflight_prefills.discard(request)
 1119         request.status = RequestStatus.PREEMPTED
 1120         request.num_computed_tokens = 0
 1121         if request.spec_token_ids:
 1122             request.spec_token_ids = []
 1123         request.num_preemptions += 1
 1124         if self.log_stats:
 1125             request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
 1126 
 1127         # Put the request back to the waiting queue.
 1128         self.waiting.prepend_request(request)
 1129 
 1130     def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
 1131         # Advance the number of computed tokens for the request AFTER
 1132         # the request is scheduled.
```

## vllm/v1/core/kv_cache_manager.py

SHA256: `2e91e77f84dfd9d4bed79f50392be72559207f54e845a3d3402baf88dc529320`

原文件 360–425 行：

```text
  360             self.max_model_len,
  361         )
  362 
  363         watermark_blocks = 0
  364         # The watermark is applied to waiting/preempted requests only, and only
  365         # when there's at least one request already scheduled.
  366         if has_scheduled_reqs and request.status in (
  367             RequestStatus.WAITING,
  368             RequestStatus.PREEMPTED,
  369         ):
  370             watermark_blocks = self.watermark_blocks
  371 
  372         if full_sequence_must_fit:
  373             # First check and fail if the full request sequence won't fit.
  374             full_num_tokens = min(request.num_tokens, self.max_model_len)
  375 
  376             num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
  377                 request_id=request.request_id,
  378                 num_tokens=full_num_tokens,
  379                 new_computed_blocks=new_computed_block_list,
  380                 num_encoder_tokens=num_encoder_tokens,
  381                 total_computed_tokens=total_computed_tokens,
  382                 num_tokens_main_model=full_num_tokens,
  383                 apply_admission_cap=True,
  384             )
  385             required_blocks = num_blocks_to_allocate + watermark_blocks
  386             if required_blocks > self.block_pool.get_num_free_blocks():
  387                 return None
  388 
  389         num_tokens_main_model = total_computed_tokens + num_new_tokens
  390         num_tokens_need_slot = min(
  391             num_tokens_main_model + num_lookahead_tokens, self.max_model_len
  392         )
  393 
  394         # Free the blocks that are skipped during the attention computation
  395         # (e.g., tokens outside the sliding window).
  396         # We can do this even if we cannot schedule this request due to
  397         # insufficient free blocks.
  398         # Should call this function before allocating new blocks to reduce
  399         # the number of evicted blocks.
  400         self.coordinator.remove_skipped_blocks(
  401             request.request_id, total_computed_tokens
  402         )
  403 
  404         num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
  405             request_id=request.request_id,
  406             num_tokens=num_tokens_need_slot,
  407             new_computed_blocks=new_computed_block_list,
  408             num_encoder_tokens=num_encoder_tokens,
  409             total_computed_tokens=num_local_computed_tokens
  410             + num_external_computed_tokens,
  411             num_tokens_main_model=num_tokens_main_model,
  412         )
  413 
  414         # Keep `reserved_blocks` free for other in-flight sequences, and an
  415         # additional watermark of headroom for waiting/preempted admissions.
  416         available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
  417         required_blocks = num_blocks_to_allocate + watermark_blocks
  418         if required_blocks > available_blocks:
  419             # Cannot allocate new blocks
  420             return None
  421 
  422         if (
  423             new_computed_block_list is not self.empty_kv_cache_blocks.blocks
  424             or num_external_computed_tokens > 0
  425         ):
```

## vllm/config/scheduler.py

SHA256: `4102ccd3d73188200b25f47a0cba837853662647850f283012d8cc1f9e8c47c2`

原文件 140–151 行：

```text
  140     scheduler_reserve_full_isl: bool = True
  141     """If True, the scheduler checks whether the full input sequence length
  142     fits in the KV cache before admitting a new request, rather than only
  143     checking the first chunk. Prevents over-admission and KV cache thrashing
  144     with chunked prefill."""
  145 
  146     watermark: float = Field(default=0.0, ge=0.0, lt=1.0)
  147     """Fraction of total KV cache blocks to keep free (the watermark) when
  148     admitting waiting or preempted requests into the running queue. This headroom
  149     helps avoid frequent KV cache eviction and the resulting repeated preemption
  150     of requests when GPU memory is scarce. Must be in the range [0.0, 1.0); 0.0
  151     (the default) disables the watermark."""
```

