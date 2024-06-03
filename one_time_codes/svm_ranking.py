from sklearn.svm import SVC
import numpy as np

# Assume we have 10 items, so we need 10-dimensional binary indicator vectors for each
num_items = 10
# Generate binary indicator vectors
indicator_vectors = np.eye(num_items)

# Simulate pairwise comparisons
# For simplicity, let's randomly decide which item wins in each comparison
np.random.seed(42)
comparisons = []
labels = []
for i in range(num_items):
    for j in range(i + 1, num_items):
        comparison_vector = indicator_vectors[i] - indicator_vectors[j]
        # Randomly decide the winner
        winner = np.random.choice([1, -1])
        comparisons.append(comparison_vector)
        labels.append(winner)

# Convert lists to numpy arrays
comparisons = np.array(comparisons)
labels = np.array(labels)

# Train SVM to learn ranking
svm = SVC(kernel='linear', C=1.0)
svm.fit(comparisons, labels)

# The ranking is inferred from the weight vector of the trained SVM
w = svm.coef_[0]

# Normally, you would use the SVM to predict comparisons between items
# For ranking, we sort items based on their weights in w
# Higher weight means higher rank
ranked_indices = np.argsort(w)[::-1]  # Indices of items sorted by weight, highest first

print("Ranked items (0-indexed):", ranked_indices)
