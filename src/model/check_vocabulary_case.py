from src.model.load_model import load_model

W, b, vocabulary = load_model()

print("but:", vocabulary.get("but"))
print("But:", vocabulary.get("But"))

uppercase_words = [
    word for word in vocabulary
    if word != word.lower()
]

print("\nTotal uppercase vocabulary words:", len(uppercase_words))
print("\nFirst 30 uppercase words:")

for word in uppercase_words[:30]:
    print(word)