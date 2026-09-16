from datasets import load_dataset
import json

LABEL_TO_ANSWER = {0: "Yes", 1: "Unclear", 2: "No"}


def build_row(data, count):
    user_content = f"{data['premise']} \nQuestion: {data['hypothesis']} Yes, no or unclear?\nAnswer:"
    assistant_content = LABEL_TO_ANSWER[data['label']]
    return {
        "dataset": "anli",
        "id": count,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


train_df = load_dataset('facebook/anli')['train_r1']
test_df = load_dataset('facebook/anli')['test_r1']

count = 0
with open('/net/spaces/scratch/lcpandia/data/processed/facebook/anli/r1/test_anli.jsonl', 'w') as test_anli:
    for data in test_df:
        json.dump(build_row(data, count), test_anli)
        test_anli.write('\n')
        count += 1

count = 0
with open('/net/spaces/scratch/lcpandia/data/processed/facebook/anli/r1/train_anli.jsonl', 'w') as train_anli:
    for data in train_df:
        json.dump(build_row(data, count), train_anli)
        train_anli.write('\n')
        count += 1
