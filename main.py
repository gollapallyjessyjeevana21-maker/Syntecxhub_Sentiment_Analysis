import re
import math
from collections import Counter

# Sample labelled dataset
data = [
    ("i love this product", "positive"),
    ("this is amazing", "positive"),
    ("i am very happy", "positive"),
    ("excellent service and great quality", "positive"),
    ("this is wonderful", "positive"),
    ("i really like it", "positive"),
    ("the experience was fantastic", "positive"),
    ("this product is useful", "positive"),
    ("i hate this product", "negative"),
    ("this is terrible", "negative"),
    ("i am very disappointed", "negative"),
    ("poor service and bad quality", "negative"),
    ("this is awful", "negative"),
    ("i really dislike it", "negative"),
    ("the experience was horrible", "negative"),
    ("this product is useless", "negative")
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

# Training and testing data
train_data = data[:12]
test_data = data[12:]

classes = ["positive", "negative"]
word_counts = {c: Counter() for c in classes}
class_counts = Counter()
vocabulary = set()

# Train the Naive Bayes model
for text, label in train_data:
    words = clean_text(text)
    class_counts[label] += 1
    word_counts[label].update(words)
    vocabulary.update(words)

def predict(text):
    words = clean_text(text)
    scores = {}

    for label in classes:
        total_words = sum(word_counts[label].values())
        score = math.log(class_counts[label] / len(train_data))

        for word in words:
            count = word_counts[label][word]
            score += math.log(
                (count + 1) / (total_words + len(vocabulary))
            )

        scores[label] = score

    return max(scores, key=scores.get)

# Evaluate model performance
correct = 0
tp = fp = fn = 0

for text, actual in test_data:
    predicted = predict(text)

    if predicted == actual:
        correct += 1

    if predicted == "positive" and actual == "positive":
        tp += 1
    elif predicted == "positive" and actual == "negative":
        fp += 1
    elif predicted == "negative" and actual == "positive":
        fn += 1

accuracy = correct / len(test_data)
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

f1_score = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

print("SENTIMENT ANALYSIS TOOL")
print("=" * 30)
print("Test Accuracy:", round(accuracy * 100, 2), "%")
print("F1 Score:", round(f1_score, 2))

# Sample predictions
reviews = [
    "I love this product",
    "This is terrible",
    "The service is excellent",
    "I hate this experience"
]

print("\nSAMPLE PREDICTIONS")

for review in reviews:
    print("\nReview:", review)
    print("Predicted Sentiment:", predict(review).upper())
