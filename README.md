Product Classification using SVM with SIFT and KMeans Clustering
Overview
This project aims to classify product images into different categories using a combination of computer vision and machine learning techniques. The key steps involved are feature extraction using SIFT (Scale-Invariant Feature Transform), clustering of features using KMeans, and classification using a Support Vector Machine (SVM) model. The project also includes hyperparameter tuning using GridSearchCV and performance evaluation.

Dependencies
The following Python libraries are required to run the code:

OpenCV (cv2)
NumPy (numpy)
OS (os)
Scikit-learn (sklearn)
Matplotlib (matplotlib)
Seaborn (seaborn)
Joblib (joblib)
You can install the required packages using pip:

bash
Copy code
pip install opencv-python-headless numpy scikit-learn matplotlib seaborn joblib
Project Structure
Dataset: The dataset is expected to be organized into folders, with each product category having its own subfolder. Inside each product category folder, there should be two subfolders: Train and Validation, containing the training and validation images, respectively.

python
Copy code
Product Classification/
├── Product_Category_1/
│   ├── Train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── Validation/
│       ├── image3.jpg
├── Product_Category_2/
│   ├── Train/
│   ├── Validation/
...
Code: The Python script implements the entire pipeline from loading images to classification and evaluation.

How to Run
Load Images: The script loads images from the specified dataset folder and extracts labels based on the folder names.

Feature Extraction (SIFT): The sift_extraction() function extracts SIFT features from each image.

Clustering (KMeans): The clusters() function clusters the extracted features using KMeans to create a visual vocabulary.

Visual Word Descriptor: The visual_word_descriptor() function assigns each feature to the nearest visual word.

Histogram Generation: The generate_histogram() function creates histograms of visual word occurrences for each image.

Classification (SVM):

An SVM classifier pipeline is created using make_pipeline() with standardization and the SVM model.
Cross-validation is performed on the validation set to assess accuracy.
Hyperparameter tuning is conducted using GridSearchCV.
The model is trained on the entire training set, and performance is evaluated on both the training and validation sets.
Model Saving: The trained model is saved as svm_classifier_model.joblib for future use.

Performance Metrics:

Accuracy, classification reports, and confusion matrices are generated to evaluate model performance.
Training and testing times are measured.
Confusion Matrix: The confusion matrix is plotted to visualize the performance of the classifier.

Hyperparameter Tuning
GridSearchCV is used to tune the following SVM hyperparameters:

C: Regularization parameter
kernel: Kernel type (linear, rbf, sigmoid)
Results
The results of the model, including cross-validation scores, training and validation accuracy, classification reports, and confusion matrices, are printed to the console.

Notes
Ensure the dataset is properly organized into training and validation sets before running the script.
Adjust the number of clusters (k) in KMeans as needed. The script currently uses k=500.
The script suppresses specific warnings related to image profiles for cleaner output.
