import json

import pandas as pd

SYSTEM_PROMPT = "You are a helpful assistant. You provide the user with the answer."

USER_TEMPLATE = """Your task is to classify each product as being an Exact, Substitute, Complement, or Irrelevant match for the query.

Here is the user's query:
{user_query}

Here is the product information:
{product_info}

--------------

Your final response must be within <answer> </answer> tags. Label the relation type as a number: 0 = Exact, 1 = Substitute, 2 = Complement and 3 = Irrelevant. For example, <answer>one of [0, 1, 2, 3]</answer>."""

ASSISTANT_PREFIX = "<answer>"

LABEL_TO_ANSWER = {"E": 0, "S": 1, "C": 2, "I": 3}


def format_product_info(row):
    fields = [
        ("Title", row.product_title),
        ("Brand", row.product_brand),
        ("Color", row.product_color),
        ("Bullet Points", row.product_bullet_point),
        ("Description", row.product_description),
    ]
    lines = [f"{name}: {value}" for name, value in fields if pd.notna(value) and str(value).strip()]
    return "\n".join(lines)


def build_row(row, count):
    user_content = USER_TEMPLATE.format(user_query=row.query, product_info=format_product_info(row))
    assistant_content = f"{ASSISTANT_PREFIX}{LABEL_TO_ANSWER[row.esci_label]}</answer>"
    return {
        "dataset": "esci",
        "id": count,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def write_jsonl(df, path):
    with open(path, "w") as f:
        for count, row in enumerate(df.itertuples()):
            f.write(json.dumps(build_row(row, count)) + "\n")


LABEL_NAMES = {"E": "Exact", "S": "Substitute", "I": "Irrelevant", "C": "Complement"}
# Column order matches the paper's Table 3 (Exact, Substitute, Irrelevant, Complement),
# not LABEL_TO_ANSWER's (Exact, Substitute, Complement, Irrelevant) numeric assignment.
TABLE_COLUMN_ORDER = ["E", "S", "I", "C"]


def print_label_distribution(named_splits):
    """named_splits: [(display_name, df), ...], e.g. [("Train (49K)", df_train_sample), ...].
    Reproduces the paper's Table 3 layout: count (percentage%) per label per split."""
    header = f"{'Split':<14}" + "".join(f"{LABEL_NAMES[label]:>18}" for label in TABLE_COLUMN_ORDER)
    print(header)
    for name, df in named_splits:
        n = len(df)
        counts = df["esci_label"].value_counts()
        row = f"{name:<14}"
        for label in TABLE_COLUMN_ORDER:
            count = int(counts.get(label, 0))
            pct = 100 * count / n if n else 0.0
            cell = f"{count:,} ({pct:.2f}%)"
            row += f"{cell:>18}"
        print(row)


df_examples = pd.read_parquet('shopping_queries_dataset_examples.parquet')
df_products = pd.read_parquet('shopping_queries_dataset_products.parquet')
df_sources = pd.read_csv("shopping_queries_dataset_sources.csv")

df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale','product_id'],
    right_on=['product_locale', 'product_id']
)

df_task_2 = df_examples_products[df_examples_products["large_version"] == 1]
df_task_2 = df_task_2[df_task_2["product_locale"] == "us"]
df_task_2_train = df_task_2[df_task_2["split"] == "train"]
df_task_2_test = df_task_2[df_task_2["split"] == "test"]

SEED = 123
TRAIN_SAMPLE_SIZE = 50_000
TEST_SAMPLE_SIZE = 10_000
VAL_HOLDOUT_SIZE = 1_000

df_train_sample = df_task_2_train.sample(n=TRAIN_SAMPLE_SIZE, random_state=SEED)
df_test_sample = df_task_2_test.sample(n=TEST_SAMPLE_SIZE, random_state=SEED)

# Hold out 1K of the 50K training sample as validation, leaving 49K for training.
df_val_sample = df_train_sample.sample(n=VAL_HOLDOUT_SIZE, random_state=SEED)
df_train_sample = df_train_sample.drop(df_val_sample.index)

print_label_distribution([
    (f"Train ({len(df_train_sample) // 1000}K)", df_train_sample),
    (f"Val ({len(df_val_sample) // 1000}K)", df_val_sample),
    (f"Test ({len(df_test_sample) // 1000}K)", df_test_sample),
])

write_jsonl(df_train_sample, '/net/spaces/scratch/lcpandia/data/processed/esci/train_esci.jsonl')
write_jsonl(df_val_sample, '/net/spaces/scratch/lcpandia/data/processed/esci/val_esci.jsonl')
write_jsonl(df_test_sample, '/net/spaces/scratch/lcpandia/data/processed/esci/test_esci.jsonl')
