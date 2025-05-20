# AI-Based Educational Question Classification (Bloom’s Taxonomy)
This project leverages Natural Language Processing (NLP) and machine learning techniques to automatically classify educational questions based on Bloom’s Taxonomy — a widely used framework for categorizing educational objectives by cognitive complexity.

# 📌 Objective
Automatically categorize educational questions into Bloom’s Taxonomy levels:

Remember

Understand

Apply

Analyze

Evaluate

Create

This classification can enhance adaptive learning systems, streamline content tagging, and support intelligent tutoring.

# 🛠️ Technologies Used
Python 3.9+

Scikit-learn

Word2Vec (Gensim)

TF-IDF (TfidfVectorizer)

NumPy / Pandas / Matplotlib / Seaborn

Jupyter Notebook

# 📊 Models Trained
Support Vector Machine (SVM - Linear & RBF)

Random Forest

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

# 🧠 Feature Engineering
Word2Vec Embeddings: Captures semantic meaning.

TF-IDF Vectorization: Emphasizes term relevance across documents.

Text Preprocessing: Lowercasing, punctuation removal, stopword filtering, lemmatization.

# 🧪 Dataset
6,458 questions covering a broad range of topics and educational levels.

Initial 600 questions collected from educational sources and expanded using ChatGPT-based data augmentation to introduce variability and prevent overfitting.

# 🚀 Results
SVM with RBF kernel achieved the highest accuracy of 99.7%.

Evaluated using Accuracy, Precision, Recall, F1-score, and Cohen’s Kappa.

SVM models showed consistent high performance, especially with advanced preprocessing.

# 📂 Repository Structure
/BloomTaxonomyClassifier/
│
├── BloomMultipleModelsCustomized.ipynb     # Customized classification approach with tuning
├── BloomMultipleModelsEmbedding.ipynb      # Embedding-based feature engineering approach
├── AI_Cognitive_Classification_Project.pdf # Final project report
├── README.md                               # Project documentation
📈 Future Work
Extend classification to multi-label categories.

Incorporate ensemble methods for higher accuracy.

Implement user feedback loop to fine-tune predictions over time.

