# 🐱🐶 Cats vs Dogs Classification using SVM

This project implements a **Support Vector Machine (SVM)** to classify images of cats and dogs from the [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
It uses **Histogram of Oriented Gradients (HOG)** features for image representation and trains a linear SVM classifier to distinguish between cats and dogs.

---

## 🚀 Project Workflow

1. **Mount Google Drive** – Access dataset stored in Google Drive.
2. **Install Dependencies** – Ensure compatible versions of NumPy and Scikit-learn.
3. **Preprocessing**

   * Resize images to fixed size (128×128).
   * Convert RGB images to grayscale.
   * Extract HOG features.
4. **Model Training**

   * Train SVM with a linear kernel (`SVC(kernel='linear')`).
   * Use 80/20 train-test split.
5. **Evaluation**

   * Accuracy score
   * Classification report (precision, recall, f1-score)
   * Confusion matrix
6. **Model Saving** – Save trained model as `.pkl` file using `joblib`.
7. **Prediction**

   * Predict a single image.
   * Extended function to predict multiple images.

---

## 📂 Dataset

* Dataset: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
* Structure after extraction:

  ```
  train/
      cat.0.jpg
      cat.1.jpg
      dog.0.jpg
      dog.1.jpg
  test1/
      1.jpg
      2.jpg
      ...
  ```

---

## 🛠️ Installation

Run in **Google Colab**:

```bash
!pip install --upgrade --force-reinstall numpy==1.26.4 scikit-learn==1.5.2
!pip install scikit-image tqdm joblib
```

---

## 📘 Usage

### Training

```python
X, y = load_data(DATA_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
svm_clf = SVC(kernel='linear', probability=True)
svm_clf.fit(X_train, y_train)
```

### Evaluation

```python
y_pred = svm_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))
```

### Save Model

```python
import joblib
joblib.dump(svm_clf, "svm_dog_cat_hog.pkl")
```

### Predict Images

```python
print(predict_image("test1/1.jpg"))        # Single image
print(predict_image(["test1/1.jpg", "test1/2.jpg"]))  # Multiple images
```

---

## 📊 Results

* **Classifier**: Linear SVM with HOG features
* **Performance**: Accuracy depends on feature extraction & dataset subset, typically \~70–80% with HOG+SVM.
* For higher accuracy, deep learning (CNNs) is recommended.

---

## 🔮 Future Improvements

* Use **GridSearchCV** to tune SVM parameters (`C`, kernel type).
* Apply **PCA** for dimensionality reduction.
* Replace HOG+SVM with **CNN (Convolutional Neural Networks)** for improved accuracy.

---

## 📜 License

This project is for **educational purposes**. Dataset belongs to Kaggle competition organizers.


