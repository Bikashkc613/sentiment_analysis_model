
from src.model.load_model import load_model


# Step 1: Load trained model
W, b, vocabulary = load_model()

words = ["hated", "hate", "terrible", "boring",
         "wonderful", "amazing", "loved"]

print("\nVOCABULARY AND WEIGHT CHECK")

for word in words:
    if word in vocabulary:
        index = vocabulary[word]

        print(f"\nWord: {word}")
        print("Vocabulary index:", index)
        print("Weights:", W[index])
    else:
        print(f"\n{word}: NOT IN VOCABULARY")


# Step 2: Reverse vocabulary
# Example: {"movie": 0} becomes {0: "movie"}

index_to_word = {
    index: word
    for word, index in vocabulary.items()
}


# Step 3: Define sentiment classes
class_names = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# Step 4: Inspect each sentiment
for class_id, class_name in class_names.items():

    print("\n" + "=" * 60)
    print(f"TOP WORDS FOR {class_name.upper()} SENTIMENT")
    print("=" * 60)

    # Get weights for this sentiment
    class_weights = W[:, class_id]

    # Find indices of the 20 largest weights
    top_indices = class_weights.argsort()[-20:][::-1]

    for index in top_indices:

        word = index_to_word[index]
        weight = class_weights[index]

        print(f"{word:20} {weight:.6f}")
