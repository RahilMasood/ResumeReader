# ResumeReader


## **Project Overview**
This project focuses on building a machine learning model to classify resumes into predefined categories (e.g., Data Science, Web Development, etc.) based on their content. By leveraging **Natural Language Processing (NLP)** techniques and the **Random Forest Classifier**, we aim to create a robust and efficient system for resume categorization.

---

## **Dataset**
We used a dataset containing resumes in textual form and their respective categories. The dataset has two main columns:
- **`Resume`**: The raw text of the resumes.
- **`Category`**: The label indicating the category of the resume.

## **Algorithm Used**
We used the **Random Forest Classifier** for this project. The Random Forest algorithm is a robust and reliable classification technique that combines multiple decision trees to improve accuracy and prevent overfitting.


---

## **Results**
- **Cross-Validation Accuracy**: ~[0.96753247 0.95454545 0.96753247 0.97402597 0.96732026]
- **Training Accuracy**: ~99%
- **Testing Accuracy**: ~98%

The Random Forest Classifier performed well, demonstrating its ability to accurately categorize resumes into their respective categories.

---

## **How to Use**
1. **Install Dependencies**:
   - Ensure you have the following Python libraries installed: `pandas`, `scikit-learn`, `re`.
2. **Load the Dataset**:
   - Place your dataset (`ResumeDataSet.csv`) in the specified directory and ensure it contains the columns `Resume` and `Category`.
3. **Run the Script**:
   - Execute the script to clean the data, train the model, and evaluate its performance.
4. **Predict New Resumes**:
   - Modify the script to input new resumes and use the trained model to predict their categories.

---

## **Future Improvements**
1. **Deep Learning**:
   - Experiment with deep learning models like LSTMs or Transformers for improved classification.
2. **Enhanced Preprocessing**:
   - Use advanced NLP techniques like lemmatization or named entity recognition (NER) to improve text cleaning.
3. **Hyperparameter Tuning**:
   - Optimize the hyperparameters of the Random Forest model for even better performance.
4. **Deployment**:
   - Build a web interface to allow users to upload resumes and get instant predictions.

---

## **Conclusion**
This project demonstrates how machine learning and NLP techniques can automate the classification of resumes, saving time and effort in hiring processes. The Random Forest Classifier, with its robustness and accuracy, proved to be an excellent choice for this task.
