# from datasets import load_dataset
from huggingface_hub import hf_hub_download
# mmsi_bench = load_dataset("RunsenXu/MMSI-Bench")
# print(mmsi_bench)
import pandas as pd
import os

parquet_path = hf_hub_download(
    repo_id="RunsenXu/MMSI-Bench",
    filename="MMSI_Bench.parquet",
    repo_type="dataset",
)
print(parquet_path)

df = pd.read_parquet(parquet_path)

# Count the number of unique question types
num_types = df['question_type'].nunique()
print(f"\nTotal number of question types: {num_types}")

# Count samples per question type
type_counts = df['question_type'].value_counts()
print(f"\nNumber of samples per question type:")
print(type_counts)

# Print summary statistics
print(f"\nTotal number of samples: {len(df)}")
print(f"\nSummary:")
for question_type, count in type_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {question_type}: {count} samples ({percentage:.2f}%)")

# Print first three questions for each type
print(f"\n{'='*80}")
print("First 3 questions for each question type:")
print(f"{'='*80}")

for question_type in type_counts.index:
    print(f"\n{'─'*80}")
    print(f"Question Type: {question_type}")
    print(f"{'─'*80}")
    
    # Get first 3 samples of this type
    type_samples = df[df['question_type'] == question_type].head(3)
    
    for idx, row in type_samples.iterrows():
        print(f"\n[Sample {idx}]")
        print(f"  ID: {row['id']}")
        print(f"  Question: {row['question']}")
        if 'answer' in row:
            print(f"  Answer: {row['answer']}")
        if 'thought' in row and pd.notna(row['thought']):
            print(f"  Thought: {row['thought']}")
        print()

output_dir = '/data/Datasets/MindCube/mmsi-bench/images'
os.makedirs(output_dir, exist_ok=True)

# for idx, row in df.iterrows():
#     id_val = row['id']
#     images = row['images']  
#     question_type = row['question_type']
#     question = row['question']
#     answer = row['answer']
#     thought = row['thought']

#     image_paths = []
#     if images is not None:
#         for n, img_data in enumerate(images):
#             image_path = f"{output_dir}/{id_val}_{n}.jpg"
#             with open(image_path, "wb") as f:
#                 f.write(img_data)
#             image_paths.append(image_path)
#     else:
#         image_paths = []

#     print(f"id: {id_val}")
#     print(f"images: {image_paths}")
#     print(f"question_type: {question_type}")
#     print(f"question: {question}")
#     print(f"answer: {answer}")
#     print(f"thought: {thought}")
#     print("-" * 50)


