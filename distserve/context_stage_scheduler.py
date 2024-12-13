from abc import ABC, abstractmethod
import copy
from typing import List, Callable, Tuple

from distserve.config import ContextStageSchedConfig, ParallelConfig
from distserve.logger import init_logger
from distserve.request import Request, BatchedRequests, MigratingRequest, MigratingRequest2
from distserve.block_manager import BlockManager

logger = init_logger(__name__)


class ContextStageScheduler(ABC):
    """
    ContextStageScheduler: The abstract class for a context scheduler.
    
    It should maintain all the requests in the current systems, and support two basic ops:
        - add_request: Add a newly arrived request into the waiting queue
        - get_next_batch_and_pop: Get the next batch for the context stage, and 
          pop the requests in the batch from the waiting queue.
    
    This scheduler is much simpler than DecodingStageScheduler since one request
    will only be processed by one context stage.      
    """

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """
        Add a request to the scheduler.
        """
        raise NotImplementedError()

    @abstractmethod
    def abort_request(self, request_id: int) -> None:
        """
        Cancel a request from the scheduler.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        Get a batch of requests for the execution of next iteration and
        pop the requests in the batch from the waiting queue.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_waiting_requests(self) -> int:
        """
        Get the number of requests that are waiting for processing.
        """
        raise NotImplementedError()

    @abstractmethod
    def print_status(self) -> None:
        """
        Print the status of the scheduler.
        """
        raise NotImplementedError()
    
    def on_finish_requests(self, batch: BatchedRequests) -> None:
        """
        Callback function when a batch of requests finish the context stage.
        """
        pass
    
    def on_request_migrated(self, migrated_request: MigratingRequest) -> None:
        """
        Callback function when a request is migrated to the decoding stage
        """
        pass
    
    def post_process(self) -> None:
        """
        Post process after each iteration. ContextEventLoop will call this
        function after each iteration.
        """
        pass


class ContextStageFCFSScheduler(ContextStageScheduler):
    """
    A first-come-first-serve scheduler.
    """

    def __init__(
        self,
        sched_config: ContextStageSchedConfig, 
        parallel_config: ParallelConfig,
        block_manager: BlockManager,
        engine_migrate_block_callback: Callable,
    ):
        
        assert (
            sched_config.policy == "fcfs"
        ), f"can not initialize a FCFS scheduler with policy {sched_config.policy}"
        self.sched_config = sched_config
        # If the current batch is full, the requests will be put into the waiting queue.
        self.waiting_queue = []

        self.migrate_queue : List[MigratingRequest2] = []        

        self.parallel_config: List[Request] = copy.deepcopy(parallel_config)
        self.block_manager = block_manager

        self.engine_migrate_block_callback = engine_migrate_block_callback
        #print(f'self.engine_migrate_block_callback:{self.engine_migrate_block_callback}')
        # Requests that finished the context stage but are not accepted by the decoding stage.
        self.unaccepted_queue: List[Request] = []
        # The number of on-the-fly (i.e. processing) request blocks
        # Adds when calling get_next_batch_and_pop()
        # Subtracts when calling on_finish_requests()
        self.num_on_fly_request_block = 0

    def add_request(self, request: Request) -> None:
        """
        Add a request to the scheduler.
        """
        self.waiting_queue.append(request)

    async def add_migrate_request(self, request:MigratingRequest2) -> None:
        self.migrate_queue.append(request)

    def abort_request(self, request_id: int) -> None:
        """
        Cancel a request from the scheduler.
        """
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return

    def _get_block_needed(self, length: int):
        block_size = self.block_manager.cache_config.block_size
        return (length + block_size - 1) // block_size
            
    def get_next_batch_and_pop(self) -> BatchedRequests:
        """
        Get the next batch for the context stage in a FCFS-like manner, and pop them
        """
        next_batch = BatchedRequests()

        def _check_add_to_cur_batch(request: Request) -> bool:
            """
            Check whether the request can be added to the current batch.
            """
            return (
                # Limit 1. batch size
                len(next_batch) < self.sched_config.max_batch_size
            ) and (
                # Limit 2. tokens per batch
                next_batch.get_num_input_tokens()
                + request.get_num_input_tokens()
                <= self.sched_config.max_tokens_per_batch
            ) and (
                # Limit 3. GPU blocks
                sum([
                    self._get_block_needed(len(req.prompt_token_ids))
                    for req in next_batch.requests + [request]
                ]) +
                sum([
                    self._get_block_needed(len(req.prompt_token_ids))
                    for req in self.unaccepted_queue
                ]) +
                self.num_on_fly_request_block 
                <= self.block_manager.max_num_gpu_blocks
            )
        
        #print(f'len(self.migrate_queue):{len(self.migrate_queue)}')

        '''while len(self.migrate_queue) > 0:
            request = self.migrate_queue[0]
            if _check_add_to_cur_batch(request):
                next_batch.add_request(request)
                self.migrate_queue.pop(0)
            else:
                break'''
 
        while len(self.waiting_queue) > 0:
            request = self.waiting_queue[0]
            if _check_add_to_cur_batch(request):
                next_batch.add_request(request)
                self.waiting_queue.pop(0)
            else:
                break
               
        #print(f'prefill_len(next_batch):{len(next_batch)}')
 
        self.num_on_fly_request_block += sum([
            self._get_block_needed(req.get_input_len())
            for req in next_batch.requests
        ])

        return next_batch

    def on_finish_requests(self, batch: BatchedRequests):
        for request in batch.requests:
            if not request.is_finished:
                self.unaccepted_queue.append(request)
        
        self.num_on_fly_request_block -= sum([
            self._get_block_needed(req.get_input_len())
            for req in batch.requests
        ])
    
    def on_request_migrated(self, migrated_request: MigratingRequest):
        for i, request in enumerate(self.unaccepted_queue):
            if request.request_id == migrated_request.req.request_id:
                del self.unaccepted_queue[i]
                return
            
    def get_num_waiting_requests(self) -> int:
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        return (
            f"FCFS(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )
    
    def print_status(self):
        logger.info(f"(context) {len(self.waiting_queue)} waiting, {len(self.unaccepted_queue)} finished but unaccepted, {self.num_on_fly_request_block} blocks occupied by on-the-fly requests")
    
  
    async def post_process(self) -> None:
        def should_accept(migrating2_req: MigratingRequest2) -> bool:
            return sum([self._get_block_needed(len(req.prompt_token_ids))
                        for req in self.waiting_queue]) \
                            < self.block_manager.max_num_gpu_blocks * self.sched_config.waiting_block_prop_threshold \
                            and self._get_block_needed(len(migrating2_req.req.prompt_token_ids)) <= self.block_manager.get_num_avail_gpu_blocks()
           
        #print(f'len(self.migrate_queue):{len(self.migrate_queue)}')  
        while len(self.migrate_queue) > 0:
            migrating2_req = self.migrate_queue[0]
            if should_accept(migrating2_req):
                #print(f'len(self.migrate_queue):{len(self.migrate_queue)}')
                self.migrate_queue.pop(0)
                #print(f'len(self.migrate_queue):{len(self.migrate_queue)}')
                print(f'migrating2_req.req.request_id:{migrating2_req.req.request_id};migrating2_req.req.turn:{migrating2_req.req.turn}')
                await self.engine_migrate_block_callback(migrating2_req)
                self.waiting_queue.append(migrating2_req.req)
            else:
                break   

def get_context_stage_scheduler(
    sched_config: ContextStageSchedConfig,
    parallel_config: ParallelConfig,
    block_manager: BlockManager,
    engine_migrate_block_callback: Callable,
) -> ContextStageScheduler:
    if sched_config.policy == "fcfs":
        return ContextStageFCFSScheduler(sched_config, parallel_config, block_manager, engine_migrate_block_callback)
    else:
        raise NotImplementedError(f"Unknown context scheduler policy {sched_config.policy}")
    
