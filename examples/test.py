import argparse
import asyncio
from typing import List
from distserve.single_stage_engine import StepOutput
from distserve import OfflineLLM, SamplingParams, request
from distserve.request import MigratingRequest,MigratingRequest2
from distserve.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    ContextStageSchedConfig,
    DecodingStageSchedConfig
)
from distserve.lifetime import LifetimeEvent, LifetimeEventType


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='The model to use', default='meta-llama/Llama-2-7b-hf')
args = parser.parse_args()

args.model = r'/jizhicfs/hymiezhao/my_codes/DistServe/models/gpt2-convert'
args.tokenizer = r'/jizhicfs/hymiezhao/models/gpt2'

# Sample prompts.
prompts = [
    #"Life blooms like a flower. Far away or by the road. Waiting",
    "A quick brown fox",
    #"Artificial intelligence is",
    #"To be or not to be,",
    #"one two three four"
]

#loop = asyncio.get_event_loop()

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8, top_p=0.95, max_tokens=64, stop=["\n"]
)

# Create an LLM for offline inference.
llm = OfflineLLM(
    model_config=ModelConfig(
        model=args.model,
        tokenizer=args.tokenizer
    ),
    disagg_parallel_config=DisaggParallelConfig(
        context=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        ),
        decoding=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1
        )
    ),
    cache_config=CacheConfig(
        block_size=16,
        max_num_blocks_per_req=1024,
        gpu_memory_utilization=0.9,
        cpu_swap_space=1.0
    ),
    context_sched_config=ContextStageSchedConfig(
        policy="fcfs",
        max_batch_size=1,
        max_tokens_per_batch=16384
    ),
    decoding_sched_config=DecodingStageSchedConfig(
        policy="fcfs",
        max_batch_size=1,
        max_tokens_per_batch=16384
    )
)

engine = llm.engine
context_engine = llm.engine.context_engine
context_blockmanager = context_engine.block_manager
decoding_engine = llm.engine.decoding_engine
decoding_blockmanager = decoding_engine.block_manager
def print_status():
    print(f'context_blockmanager_blocktable:{context_blockmanager.block_table}')
    print(f'decoding_blockmanager_blocktable:{decoding_blockmanager.block_table}')

async def transfer(req:MigratingRequest):
    await decoding_engine._migrate_blocks(req)

async def detransfer(req:MigratingRequest2):
    await context_engine._migrate2_blocks(req)

async def chat():
    while True:
        cur_input = input("input prompt:")
        if cur_input == "exit":
            break
        elif cur_input == "create":
            prompt = request.create_request(prompt = "hahaha", prompt_token_ids=None, sampling_params=sampling_params, request_counter=engine.request_counter, tokenizer = engine.tokenizer, arrival_time=None, request_id = 0, turn = 2)
            print(f'prompt:{prompt}')
            engine.request_outputs[prompt.request_id] = asyncio.Queue()
            engine.request_lifetime_events[prompt.request_id] = []
            engine._on_new_lifetime_event_callback(prompt.request_id, LifetimeEvent(LifetimeEventType.Issued))
        elif cur_input == "migrate":
            context_blockmanager.allocate_blocks(prompt)
            print_status()
            migrate_req = MigratingRequest(prompt,context_blockmanager.get_block_table(prompt.request_id),context_engine.parallel_config)
            await transfer(migrate_req)
            print_status()
        elif cur_input == "migrate2":
            decoding_blockmanager.allocate_blocks(prompt)
            print_status()
            migrate2_req = MigratingRequest2(prompt,decoding_blockmanager.get_block_table(prompt.request_id),decoding_engine.parallel_config)
            await detransfer(migrate2_req)
            print_status()
        else:
            continue

asyncio.run(chat())