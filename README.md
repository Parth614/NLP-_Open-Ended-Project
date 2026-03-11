# Legal Text Classification and Summarization using NLP

This project applies Natural Language Processing (NLP) techniques to analyze legal case documents from the European Court of Human Rights (ECtHR) dataset. The system performs two main tasks: multi-label classification of legal cases and extractive summarization of long legal documents.

The goal of the project is to demonstrate how classical NLP techniques and machine learning algorithms can be used to automate the analysis of complex legal texts.

------------------------------------------------------------

## Project Overview

Legal case documents are often long and difficult to analyze manually. This project builds an NLP pipeline that automatically processes legal text, predicts violated human rights articles, and generates concise summaries of cases.

The system uses text preprocessing, TF-IDF feature extraction, machine learning classification models, and extractive summarization algorithms.

------------------------------------------------------------

## Dataset

The project uses the ECtHR NAACL 2021 dataset, which contains legal case documents from the European Court of Human Rights.

Dataset features include:
- Case facts describing the legal dispute
- Violated human rights articles
- Alleged violation articles
- Rationales explaining court decisions

Dataset size:

Train Set: ~9000 cases  
Validation Set: ~1000 cases  
Test Set: ~1000 cases

------------------------------------------------------------

## Technologies Used

Python  
Pandas  
NumPy  
Scikit-learn  
NLTK  
Matplotlib  
Seaborn  
Sumy (for summarization)  
ROUGE Score  

------------------------------------------------------------

## Project Pipeline

1. Dataset Loading  
The ECtHR dataset is downloaded and loaded into pandas DataFrames.

2. Data Exploration  
Visualizations are created to analyze text length distribution and label frequencies.

3. Text Preprocessing  
- Lowercasing  
- Tokenization  
- Stopword removal  
- Lemmatization  
- Removal of special characters  

4. Feature Extraction  
TF-IDF vectorization converts text into numerical feature vectors using unigrams and bigrams.

5. Multi-Label Classification  

Three machine learning models are implemented:

Logistic Regression  
Support Vector Machine (SVM)  
Naive Bayes  

6. Text Summarization  

Three extractive summarization algorithms are implemented:

TextRank  
Latent Semantic Analysis (LSA)  
Luhn Algorithm  

7. Evaluation  

Classification Metrics:
- Micro F1 Score
- Macro F1 Score
- Samples F1 Score
- Hamming Loss

Summarization Metrics:
- ROUGE-1
- ROUGE-2
- ROUGE-L

------------------------------------------------------------

## Results

Classification Performance

Logistic Regression  → Micro F1 ≈ 0.60  
SVM (Balanced)      → Micro F1 ≈ 0.68  
Naive Bayes         → Micro F1 ≈ 0.55  

The Support Vector Machine model with class-balanced weighting achieved the best classification performance.

Summarization Performance

TextRank  → ROUGE-1 ≈ 0.11  
Luhn      → ROUGE-1 ≈ 0.11  
LSA       → ROUGE-1 ≈ 0.15  

LSA produced the most informative summaries among the tested methods.

------------------------------------------------------------

## Example Output

Input: Legal case description  

Output:
Predicted Violated Articles: Article 3, Article 6  

Generated Summary:
A short extractive summary highlighting the key facts of the case.

------------------------------------------------------------

## Future Improvements

Possible improvements include:

- Using transformer-based models such as BERT or LegalBERT
- Implementing deep learning based summarization
- Handling class imbalance using advanced sampling methods
- Developing a web-based interface for legal document analysis

------------------------------------------------------------

## How to Run

Clone the repository

git clone https://github.com/yourusername/legal-nlp-project.git

Install required libraries

pip install pandas scikit-learn nltk matplotlib seaborn sumy rouge-score kagglehub

Run the notebook in Google Colab or Jupyter Notebook.

------------------------------------------------------------

## Author

Parth Gupta  
B.Tech Computer Science  
NLP Minor Project
