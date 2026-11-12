
import re
import html


def clean_text(text: str) -> str:
    """Clean a review consistently for training and prediction."""

    # Convert HTML entities into normal characters
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation, keeping letters and numbers
    text = re.sub(r"[^\w\s]", " ", text)

    # Replace multiple whitespace characters with one space
    text = re.sub(r"\s+", " ", text)

    # Remove extra spaces
    text = text.strip()

    return text


if __name__ == "__main__":

    sample_review = "This movie was great! <br /><br /> I loved it."

    cleaned_review = clean_text(sample_review)

    print("Original review:")
    print(sample_review)

    print("\nCleaned review:")
    print(cleaned_review)

    test_words = [
        "But",
        "BUT",
        "but",
        "I LOVE this Movie!",
    ]

    print("\nCleaning tests:")

    for word in test_words:
        print(word, "->", clean_text(word))
