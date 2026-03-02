module {
  util.global private @__device_0 = #hal.device.target<"hip", [#hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384, dma_sizes = [32], workgroup_memory_bank_count = 32>>, ukernels = "none"}>]> : !hal.device
  hal.executable private @fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.default_tuning_spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384, dma_sizes = [32], workgroup_memory_bank_count = 32>>, ukernels = "none"}>) {
      hal.executable.export public @fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false, use_igemm_convolution = false>}>} {
          %cst = arith.constant 0.000000e+00 : f32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<10x96x96xbf16, #hal.descriptor_type<storage_buffer>>
          %1 = amdgpu.fat_raw_buffer_cast %0 resetOffset : memref<10x96x96xbf16, #hal.descriptor_type<storage_buffer>> to memref<10x96x96xbf16, #amdgpu.address_space<fat_raw_buffer>>
          %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<10x96x96xbf16, #hal.descriptor_type<storage_buffer>>
          %3 = amdgpu.fat_raw_buffer_cast %2 resetOffset : memref<10x96x96xbf16, #hal.descriptor_type<storage_buffer>> to memref<10x96x96xbf16, #amdgpu.address_space<fat_raw_buffer>>
          %4 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<10x96x96xbf16, #hal.descriptor_type<storage_buffer>>
          %5 = amdgpu.fat_raw_buffer_cast %4 resetOffset : memref<10x96x96xbf16, #hal.descriptor_type<storage_buffer>> to memref<10x96x96xbf16, #amdgpu.address_space<fat_raw_buffer>>
          %6 = iree_codegen.load_from_buffer %1 : memref<10x96x96xbf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<10x96x96xbf16>
          %7 = iree_codegen.load_from_buffer %3 : memref<10x96x96xbf16, #amdgpu.address_space<fat_raw_buffer>> -> tensor<10x96x96xbf16>
          %8 = tensor.empty() : tensor<10x96x96xbf16>
          %9 = tensor.empty() : tensor<10x96x96xf32>
          %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<10x96x96xf32>) -> tensor<10x96x96xf32>
          %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%6, %7 : tensor<10x96x96xbf16>, tensor<10x96x96xbf16>) outs(%10 : tensor<10x96x96xf32>) attrs =  {lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, promote_operands = [0, 1], reduction = [0, 0, 0, 2], subgroup = [0, 1, 1, 0], workgroup = [1, 32, 32, 0]}>, root_op} {
          ^bb0(%in: bf16, %in_0: bf16, %out: f32):
            %13 = arith.extf %in : bf16 to f32
            %14 = arith.extf %in_0 : bf16 to f32
            %15 = arith.mulf %13, %14 : f32
            %16 = arith.addf %out, %15 : f32
            linalg.yield %16 : f32
          } -> tensor<10x96x96xf32>
          %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11 : tensor<10x96x96xf32>) outs(%8 : tensor<10x96x96xbf16>) {
          ^bb0(%in: f32, %out: bf16):
            %13 = arith.truncf %in : f32 to bf16
            linalg.yield %13 : bf16
          } -> tensor<10x96x96xbf16>
          iree_codegen.store_to_buffer %12, %5 : tensor<10x96x96xbf16> into memref<10x96x96xbf16, #amdgpu.address_space<fat_raw_buffer>>
          return
        }
      }
    }
  }
  util.global private mutable @fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_rocm_hsaco_fb_fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32_buffer : !hal.buffer
  util.initializer {
    %device, %queue_affinity = hal.device.resolve on(#hal.device.affinity<@__device_0>) : !hal.device, i64
    %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
    %memory_type = hal.memory_type<"DeviceVisible|DeviceLocal"> : i32
    %buffer_usage = hal.buffer_usage<"TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage"> : i32
    %c552960 = arith.constant 552960 : index
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%queue_affinity) type(%memory_type) usage(%buffer_usage) : !hal.buffer{%c552960}
    util.global.store %buffer, @fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_rocm_hsaco_fb_fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32_buffer : !hal.buffer
    util.return
  }
  util.func public @fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_rocm_hsaco_fb_fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %0 = arith.index_cast %arg0 : i32 to index
    %device, %queue_affinity = hal.device.resolve on(#hal.device.affinity<@__device_0>) : !hal.device, i64
    %cmd = hal.command_buffer.create device(%device : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) affinity(%queue_affinity) : !hal.command_buffer
    %fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_rocm_hsaco_fb_fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32_buffer = util.global.load @fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_rocm_hsaco_fb_fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32_buffer : !hal.buffer
    %c0 = arith.constant 0 : index
    %c184320 = arith.constant 184320 : index
    %c368640 = arith.constant 368640 : index
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device : !hal.device) target(@fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0::@rocm_hsaco_fb::@fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32) : index, index, index
    %exe = hal.executable.lookup device(%device : !hal.device) executable(@fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0) : !hal.executable
    %ordinal = hal.executable.export.ordinal target(@fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0::@rocm_hsaco_fb::@fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32) : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%exe : !hal.executable)[%ordinal] workgroups([%workgroup_x, %workgroup_y, %workgroup_z]) bindings([
        (%fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_rocm_hsaco_fb_fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32_buffer : !hal.buffer)[%c0, %c184320], 
        (%fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_rocm_hsaco_fb_fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32_buffer : !hal.buffer)[%c184320, %c184320], 
        (%fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_rocm_hsaco_fb_fused_op_bmm_1cafd6d3b3dde335252db1830887f64a19bf9d5e_10x96x96xbfloat16_10x96x96xbfloat16$async_dispatch_0_batch_matmul_10x96x96x96_bf16xbf16xf32_buffer : !hal.buffer)[%c368640, %c184320]
      ]) flags("None")
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device : !hal.device> affinity(%queue_affinity) wait(%1) signal(%fence) commands(%cmd) flags("None")
    %c-1_i32 = arith.constant -1 : i32
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) flags("None") : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    util.return
  }
}
