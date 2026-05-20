import os, json, random, traceback
from src.utils import load_jsonl, save_json

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_path = os.path.join(SRC, "data", "yelp_academic_dataset_review.json")
output_path = os.path.join(SRC, "data", "yelp_sample.json")

def main():
    print("Input path:", input_path)
    print("Output path:", output_path)

    if not os.path.exists(input_path):
        print("ERROR: input file does not exist.")
        return

    try:
        # load JSON Lines
        reviews = load_jsonl(input_path)
    except Exception as e:
        print("ERROR loading JSONL:")
        traceback.print_exc()
        return

    print("Loaded reviews count:", len(reviews))

    if len(reviews) == 0:
        print("No reviews found in the input file.")
        return

    k = 50
    k = min(k, len(reviews))
    print(f"Sampling {k} reviews (or fewer if file smaller).")

    try:
        sample = random.sample(reviews, k)
    except Exception as e:
        print("ERROR during sampling:")
        traceback.print_exc()
        return

    print("Sampled reviews count:", len(sample))
    print("Preview of first sampled review (keys):", list(sample[0].keys()) if sample else None)
    print("Preview text snippet:", sample[0].get("text", "")[:200])

    try:
        save_json(sample, output_path)
        print("Saved sample successfully.")
    except Exception as e:
        print("ERROR saving sample:")
        traceback.print_exc()

if __name__ == "__main__":
    main()







