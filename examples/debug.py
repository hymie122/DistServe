bool register_ipc_mem_handle2(
	std::vector<int64_t> k_cache_handle_vec,
	std::vector<int64_t> v_cache_handle_vec,
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &decoding_parallel_config,	// Generated via ParallelConfig.to_list()
	const std::vector<int64_t> &context_parallel_config
) {
	// Convert the handles to cudaIpcMemHandle_t
	const cudaIpcMemHandle_t k_cache_handle = bytes2CudaIpcMemHandle(k_cache_handle_vec);
	const cudaIpcMemHandle_t v_cache_handle = bytes2CudaIpcMemHandle(v_cache_handle_vec);

	// First we check whether the two k/v cache area overlaps
	const int64_t decoding_tp_size = decoding_parallel_config[0];
	const int64_t decoding_tp_rank = decoding_parallel_config[1];
	const int64_t decoding_pp_size = decoding_parallel_config[2];
	const int64_t decoding_pp_rank = decoding_parallel_config[3];
	const int64_t context_tp_size = context_parallel_config[0];
	const int64_t context_tp_rank = context_parallel_config[1];
	const int64_t context_pp_size = context_parallel_config[2];
	const int64_t context_pp_rank = context_parallel_config[3];

	const int64_t layers_per_decoding_worker = num_layers / decoding_pp_size;
	const int64_t heads_per_decoding_worker = num_heads / decoding_tp_size;
	const int64_t layers_per_context_worker = num_layers / context_pp_size;
	const int64_t heads_per_context_worker = num_heads / context_tp_size;

	const int64_t decoding_start_layer = decoding_pp_rank * layers_per_decoding_worker;
	const int64_t decoding_end_layer = decoding_start_layer + layers_per_decoding_worker;
	const int64_t decoding_start_head = decoding_tp_rank * heads_per_decoding_worker;
	const int64_t decoding_end_head = decoding_start_head + heads_per_decoding_worker;

	const int64_t context_start_layer = context_pp_rank * layers_per_context_worker;
	const int64_t context_end_layer = context_start_layer + layers_per_context_worker;
	const int64_t context_start_head = context_tp_rank * heads_per_context_worker;
	const int64_t context_end_head = context_start_head + heads_per_context_worker;

	if (decoding_end_layer <= context_start_layer || decoding_start_layer >= context_end_layer ||
		decoding_end_head <= context_start_head || decoding_start_head >= context_end_head) {
		// No overlap
		return false;
	} else {
		// Overlap
		// Register the handle
		// On some platforms (e.g. non-nvlink platform) it's impossible to enable GPU p2p access, which 
		// leads to error when calling cudaIpcOpenMemHandle.
		const int64_t decoding_worker_hash = (decoding_pp_rank<<6) + decoding_tp_rank;
		cudaError_t err = cudaIpcOpenMemHandle(&decoding_worker_k_cache_addr[decoding_worker_hash], k_cache_handle, cudaIpcMemLazyEnablePeerAccess);
		if (err == cudaErrorPeerAccessUnsupported) {
			printf("Error: Peer-to-peer access is unsupported on this platform.\n");
			printf("In the current version of distserve, it is necessary to use a platform that supports GPU P2P access.\n");
			printf("Exiting...");
			exit(1);
		}
		CUDA_CHECK(cudaIpcOpenMemHandle(&decoding_worker_v_cache_addr[decoding_worker_hash], v_cache_handle, cudaIpcMemLazyEnablePeerAccess));
		return true;
	}
}