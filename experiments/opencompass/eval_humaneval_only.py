from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets

datasets = humaneval_datasets

summarizer = dict(dataset_abbrs=["openai_humaneval"])

from opencompass.models import VLLMwithChatTemplate

model_path = "/home/PC/SERE/calibration/output/qwen2_moe_similarity"

model_kwargs = dict(
    tensor_parallel_size=1,
    gpu_memory_utilization=0.80,
    trust_remote_code=True,
    max_model_len=8192,
    hf_overrides={
        "architectures": ["Qwen2MoeForCausalLMSERE"],
        "select_top_k": 2,
        "threshold": 0.1,
    },
)

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr="qwen2-moe-sere-humaneval",
        path=model_path,
        model_kwargs=model_kwargs,
        generation_kwargs=dict(temperature=0.0, top_p=1.0),
        max_out_len=1024,
        batch_size=4,
        run_cfg=dict(num_gpus=1),
    )
]

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=4),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLEvalTask),
    ),
)
