"""
Convert retaining-by-doing's data/sft/mmlu.jsonl into the format
open_instruct/dataset_transformation.py's sft_tulu_tokenize_and_truncate_v1
expects.

Same issue as the IFEval SFT data (see convert_retaining_by_doing_ifeval_sft.py):
retaining-by-doing stores the target completion in a separate `output_text`
field rather than appending it to `messages` as an assistant turn -- its own
MMLUSFTDataset does that append at load time (core/data.py:1295). Fed
directly into open-instruct, every row's `messages` is just a lone user turn,
so mask_labels() masks 100% of tokens and sft_tulu_filter_v1 drops every row,
leaving an empty dataset ("No examples left after transformation").

This script keeps every input row as-is (including rows where `correct` is
False) -- it only reshapes `messages`/`output_text` into the format
open-instruct expects. If you want a correctness filter, apply it separately
before or after running this script.

Usage:
    python scripts/data/convert_retaining_by_doing_mmlu_sft.py \
        --input data/sft/mmlu.jsonl \
        --output data/sft/mmlu_open_instruct_format.jsonl
"""
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sft/mmlu.jsonl")
    parser.add_argument("--output", default="data/sft/mmlu_open_instruct_format.jsonl")
    args = parser.parse_args()

    rows_in = [json.loads(line) for line in open(args.input)]
    rows_out = []
    num_incorrect = 0
    num_empty_output = 0
    for row in rows_in:
        if not row["correct"]:
            num_incorrect += 1
        output_text = (row.get("output_text") or "").strip()
        if not output_text:
            num_empty_output += 1
        messages = row["messages"] + [{"role": "assistant", "content": output_text}]
        rows_out.append({"messages": messages})

    with open(args.output, "w") as f:
        for row in rows_out:
            f.write(json.dumps(row) + "\n")

    print(f"input rows:                    {len(rows_in)}")
    print(f"output rows (none dropped):    {len(rows_out)}")
    print(f"  of which correct=False:      {num_incorrect}")
    print(f"  of which empty output_text:  {num_empty_output}")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
