#  Machine Learning & NLP Projects README

This repository contains **three end-to-end machine learning and NLP tasks**, demonstrating practical skills in **transformer-based NLP**, **LLM-powered text classification**, and **production-ready ML pipelines**. Each task focuses on real-world datasets, modern ML techniques, and deployment best practices.

---

##  Task 1: News Topic Classifier Using BERT

###  Problem Statement & Objective

News platforms handle a massive volume of articles daily, making manual categorization inefficient and error-prone. The objective of this task is to **automatically classify news headlines/articles into predefined topic categories** using a transformer-based model.

###  Objective of the Task

* Fine-tune a pre-trained **BERT (bert-base-uncased)** model for multi-class news classification
* Accurately classify news into one of four categories: **World, Sports, Business, Sci/Tech**
* Evaluate the model using **Accuracy and F1-score**
* Deploy the trained model for **real-time predictions** using Gradio

###  Methodology / Approach

1. **Dataset**: AG News Dataset (Hugging Face)

   * Train set: 120,000 samples
   * Test set: 7,600 samples
2. **Preprocessing**:

   * Combined title and description
   * Tokenized text using BERT tokenizer with padding and truncation
3. **Model Training**:

   * Fine-tuned `bert-base-uncased` for 3 epochs
   * Used Hugging Face `Trainer` API
4. **Evaluation Metrics**:

   * Accuracy
   * Weighted F1-score
5. **Deployment**:

   * Saved trained model and tokenizer
   * Built a Gradio web interface for live predictions

###  Key Results / Observations

* **Accuracy**: 94.84%
* **F1-Score**: 94.84%
* The fine-tuned BERT model significantly outperformed traditional ML baselines
* The model demonstrated strong semantic understanding beyond keyword matching
* Successfully deployed as an interactive web application

---

##  Task 2: End-to-End ML Pipeline with Scikit-learn

###  Problem Statement & Objective

Customer churn prediction is critical for telecom companies to reduce revenue loss. This task focuses on building a **robust, reusable, and production-ready ML pipeline** to predict customer churn.

###  Objective of the Task

* Build an end-to-end ML pipeline using **Scikit-learn Pipeline API**
* Automate preprocessing, training, and prediction
* Tune model hyperparameters efficiently
* Export the final pipeline for reuse in production environments

###  Methodology / Approach

1. **Dataset**: Telco Customer Churn Dataset
2. **Preprocessing Pipeline**:

   * Numerical feature scaling (StandardScaler)
   * Categorical feature encoding (One-Hot Encoding)
3. **Model Training**:

   * Logistic Regression
   * Random Forest Classifier
4. **Hyperparameter Tuning**:

   * Used `GridSearchCV` for optimal parameter selection
5. **Model Export**:

   * Saved complete pipeline using `joblib`

###  Key Results / Observations

* Pipelines ensured **clean, maintainable, and reproducible workflows**
* GridSearchCV improved model generalization and performance
* Exported pipeline enabled easy deployment without retraining
* Demonstrated production-ready ML engineering practices

---

##  Task 3: Auto Tagging Support Tickets Using LLM

###  Problem Statement & Objective

Support teams handle thousands of free-text tickets daily. Manual tagging is slow and inconsistent. This task aims to **automatically tag support tickets using Large Language Models (LLMs)**.

###  Objective of the Task

* Automatically assign **top 3 most probable tags** to each support ticket
* Compare **zero-shot**, **few-shot**, and **fine-tuned** LLM performance
* Improve classification accuracy using prompt engineering and learning strategies

###  Methodology / Approach

1. **Dataset**: Free-text Support Ticket Dataset
2. **Zero-shot Learning**:

   * Used a pre-trained LLM without task-specific training
3. **Few-shot Learning**:

   * Provided limited labeled examples per category in the prompt
4. **Fine-tuning**:

   * Fine-tuned a pre-trained language model on labeled tickets
5. **Multi-class Ranking**:

   * Generated and ranked probabilities for multiple tags per ticket

###  Key Results / Observations

* Zero-shot models performed reasonably but lacked domain precision
* Few-shot learning significantly improved tagging accuracy
* Fine-tuned models achieved the best performance and consistency
* Top-3 tag ranking improved flexibility and real-world usability
* Demonstrated effective use of LLMs for real-world text classification

---

##  Skills Gained Across All Tasks

* Transformer-based NLP (BERT, LLMs)
* Fine-tuning and transfer learning
* Prompt engineering and few-shot learning
* Multi-class and ranked predictions
* Scikit-learn pipelines and GridSearchCV
* Model evaluation and deployment
* Production-ready ML practices

---

##  Conclusion

These projects collectively showcase **end-to-end machine learning expertise**, from data preprocessing and model training to evaluation and deployment. They demonstrate practical, real-world applications of **modern NLP and ML techniques** suitable for academic, research, and industry settings.
