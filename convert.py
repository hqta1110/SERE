import json, pathlib

src = pathlib.Path("/home/PC/new-efficient-moe/data/lcb_calib_reap.jsonl")
dst = pathlib.Path("/home/PC/new-efficient-moe/outputs/lcb_bench_runs/lcb_prompts.jsonl")

with src.open() as fin, dst.open("w") as fout:
    for line in fin:
        row = json.loads(line)
        # Extract user message content as the prompt
        prompt = row["messages"][0]["content"]
        fout.write(json.dumps({"prompt": prompt}) + "\n")

print(f"Written {sum(1 for _ in dst.open())} prompts to {dst}")