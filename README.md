# Obesity Prediction Model -

This project focuses on building a robust machine learning model to predict obesity levels based on various lifestyle, dietary, and demographic features. The goal was to explore, preprocess, and optimize multiple classification models and select the best-performing one for deployment using **Streamlit**.

---

## **Project Overview**

1. **Objective**:  
   Develop and deploy a machine learning model capable of predicting obesity levels based on user inputs such as age, height, weight, dietary habits, physical activity, and lifestyle choices.

2. **Dataset**:  
   The dataset contains 17 features, including demographic information (e.g., age, gender), health-related habits (e.g., smoking, exercise, water consumption), and lifestyle choices (e.g., tech usage time, food between meals).  
   **Features Include**:
   - **Numerical Features**: Age, height (in meters), weight (in kg), vegetable consumption, water consumption, etc.  
   - **Categorical Features**: Gender, family history with overweight, food between meals, alcohol intake, transportation used, etc.  
   - **Target Variable**: Obesity Level (categorized as Normal, Overweight, Obesity Type I, Obesity Type II, etc.).

3. **Tools and Libraries**:  
   - **Python**: For data preprocessing, analysis, and model building.
   - **Scikit-learn**: For hyperparameter optimization and model evaluation.
   - **Streamlit**: For deploying an interactive and user-friendly web application.
   - **Pandas, NumPy, Matplotlib, Seaborn**: For data handling, exploration, and visualization.

---

## **Workflow and Steps**

### 1. **Data Exploration and Preprocessing**
- Examined the dataset for outliers, distributions, and relationships between features and target labels.
- Encoded categorical features using appropriate techniques (e.g., one-hot encoding, label encoding).
- Handled numerical features like height and weight with proper formatting (limited decimals) and calculated BMI to enhance interpretability.
- Scaled and normalized features where necessary for better model performance.

### 2. **Feature Engineering**
- **BMI Calculation**: Introduced BMI as an additional feature for better insight into user health.  
  \( \text{BMI} = \frac{\text{Weight (kg)}}{\text{Height (m)}^2} \)
- Ensured usability of input fields with appropriate min, max values, and validations in the Streamlit app.

### 3. **Model Selection and Hyperparameter Tuning**
- Tested three machine learning models:  
  - **Decision Tree Classifier**  
  - **Random Forest Classifier**  
  - **Gradient Boosting Classifier**  
- Used **GridSearchCV** to perform hyperparameter optimization with 5-fold cross-validation for each model.
- Key hyperparameters tuned for each model:
  - Decision Tree: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`
  - Gradient Boosting: `learning_rate`, `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`

### 4. **Model Evaluation**
- **Decision Tree Classifier**: Achieved 94.61% accuracy during cross-validation but prone to overfitting.
- **Random Forest Classifier**: Improved accuracy to 95.62%, with better generalization due to bagging.
- **Gradient Boosting Classifier**: Outperformed other models with a cross-validation accuracy of 97.10%, excelling at capturing complex relationships in the data through sequential error correction.

### 5. **Model Deployment**
- Selected **Gradient Boosting Classifier** as the final model due to its superior performance.
- Integrated the model into a **Streamlit** app for an interactive and user-friendly experience.
- Key features of the app:
  - Dynamic user inputs for features like age, height, weight, exercise frequency, and dietary habits.
  - Real-time BMI calculation displayed alongside predictions.
  - Responsive design with clear instructions for ease of use.

---

## **Key Findings**
1. **Model Performance**: Gradient Boosting consistently delivered the highest accuracy, making it the best choice for deployment. 
2. **Feature Importance**: Features such as BMI, exercise frequency, and calorie intake were significant predictors of obesity levels.
3. **User-Friendly Deployment**: Streamlit facilitated the creation of an intuitive web app, making the model accessible to end-users.

---

## **Technologies Used**
- **Programming Language**: Python  
- **Libraries**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit  
- **Model Optimization**: GridSearchCV  

---

## **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/widchy95/Obesity-Prediction_ML_Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Obesity-Prediction_ML_Project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---
