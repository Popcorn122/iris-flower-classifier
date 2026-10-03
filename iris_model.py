# 1. Import the tools we need
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2. Load the famous Iris dataset (built into the library)
iris = load_iris()
X = iris.data  # The flower measurements (sepal length, width, etc.)
y = iris.target # The type of flower (0, 1, or 2)

# 3. Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create the AI Model (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=3)

# 5. Train the model
print("Training the AI model on Iris data...")
model.fit(X_train, y_train)

# 6. Test the model and see how accurate it is
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"AI Model Accuracy: {accuracy * 100:.2f}%")
