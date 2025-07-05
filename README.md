# Customer Churn Prediction

## Project Domain
 
In the telecommunications industry, customer retention is a major business concern. As competition grows, companies are increasingly turning to **Machine Learning** to predict which customers are at risk of churning. By analyzing patterns in customer behavior, usage, and billing data, businesses can proactively address churn risks and improve long-term customer loyalty.

This project focuses on building a predictive model to identify customers who are most likely to leave the service. Using historical customer data, the model helps uncover key factors influencing churn decisions, such as **tenure**, **monthly charges**, and **total charges**. 

Such insights allow telecom companies like **DQLab Telco** to design targeted retention strategies, optimize resource allocation, and reduce customer turnover through early intervention.

---

##  Business Understanding

### Problem Statement
DQLab Telco, a growing telecommunications provider founded in 2019, has been focusing on customer experience from the start. Despite being a relatively young company, it has already faced a significant number of customers switching to competitors.

To address this issue, management seeks to reduce the churn rate through **machine learning-based prediction models** that can detect potential churn before it happens.

### Goals
- To build an accurate and reliable machine learning model to predict which customers are at high risk of churn.
- To provide data-driven insights regarding the key factors driving customer churn decisions.
- Ultimate goal: To reduce DQLab Telco's customer churn rate through targeted retention initiatives.

### Solution Statement
By analyzing historical customer data, we will:
- Preprocess and clean the data
- Explore and visualize patterns related to churn
- Handle class imbalance using advanced resampling techniques
- Train multiple classification models
- Select the best-performing model based on appropriate evaluation metrics

---

## Data Understanding

### Dataset Source
The dataset used in this project is provided by DQLab and can be accessed from the following link:  
[Customer Churn Prediction Dqlab Telco](https://storage.googleapis.com/dqlab-dataset/dqlab_telco.csv). This dataset contains customer data from a Telco company, including demographic details, services used, and churn information.

---

### Rows and Columns
- **7,113** rows (customers)
- **22** columns (features)

---

### Feature Descriptions

| Feature             | Description |
|---------------------|-------------|
| `UpdatedAt`         | Period of data taken |
| `customerID`        | Unique identifier for each customer |
| `gender`            | Customer's gender (Male, Female) |
| `SeniorCitizen`     | Whether the customer is a senior citizen (1 = Yes, 0 = No) |
| `Partner`           | Whether the customer has a partner (Yes, No) |
| `Dependents`        | Whether the customer has dependents (Yes, No) |
| `tenure`            | Number of months the customer has stayed with the company |
| `PhoneService`      | Whether the customer has a phone service (Yes, No) |
| `MultipleLines`     | Whether the customer has multiple lines (Yes, No, No phone service) |
| `InternetService`   | Customer’s internet service provider (DSL, Fiber optic, No) |
| `OnlineSecurity`    | Whether the customer has online security (Yes, No, No internet service) |
| `OnlineBackup`      | Whether the customer has online backup (Yes, No, No internet service) |
| `DeviceProtection`  | Whether the customer has device protection (Yes, No, No internet service) |
| `TechSupport`       | Whether the customer has tech support (Yes, No, No internet service) |
| `StreamingTV`       | Whether the customer has streaming TV (Yes, No, No internet service) |
| `StreamingMovies`   | Whether the customer has streaming movies (Yes, No, No internet service) |
| `Contract`          | The contract term of the customer (Month-to-month, One year, Two year) |
| `PaperlessBilling`  | Whether the customer has paperless billing (Yes, No) |
| `PaymentMethod`     | Customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) |
| `MonthlyCharges`    | Monthly amount charged to the customer |
| `TotalCharges`      | Total amount charged to the customer |
| `Churn`             | Whether the customer has churned (Yes, No) |

### Exploratory Data Analysis (EDA)
- **Missing & inconsistent values**: Some entries were blank and needed to be handled
- **Class imbalance**: The target variable `Churn` is imbalanced with approximately:
  - **26% churned customers**
  - **74% non-churned customers**
- **Distribution patterns**: Numerical features like `tenure`, `MonthlyCharges`, and `TotalCharges` show different distributions between churned and non-churned customers.
- **Categorical feature patterns**: Visualising **categorical features** (e.g., `Gender`, `InternetService`, `PhoneService`) using **countplots**. These visualizations helped us observe how churn is distributed across different categories.

The EDA provided valuable insights that guided the feature selection and preprocessing steps. Visual tools such as **histograms**, **boxplots**, **countplots** were extensively used to support the analysis and identify trends and anomalies.


Initial exploration revealed:
- Some missing and inconsistent values (e.g., categorical typos)
- Class imbalance: ~26% churn vs ~74% not churn

---

## Data Preparation & Data Preprocessing

A series of data cleaning and preparation steps were conducted to ensure the dataset was ready for modeling:

- **Validating Customer ID Format**: Verified that all values in the `customerID` column follow a valid a format and are unique. Invalid or duplicate IDs were removed to maintain data integrity.

- **Handling Missing Values**: Checked for missing values across all columns. Missing entries were handled by either **removing rows** with critical missing data or **imputing values** using appropriate strategies (e.g., median for numerical columns or mode for categorical ones), depending on the context and importance of each feature.

- **Handling Outliers**: Identified and treated outliers in numerical columns such as `MonthlyCharges`, `TotalCharges`, and `tenure` using the Interquartile Range (IQR) method to improve data quality.

- **Standardizing Non-Standard Categorical Values**: Ensured consistency in categorical entries by standardizing string formats (e.g., unifying values like `Male` vs `Laki-Laki`).

- **Dropping Unnecessary Columns**: Removed irrelevant features including `customerID` (a unique identifier) and `UpdatedAt` (timestamp not contributing to churn prediction).

- **Feature Encoding**: Converted categorical features into numeric format using **Label Encoding** for model compatibility.

- **Train-Test Split**: Split the dataset into training (70%) and testing (30%) subsets to evaluate model performance objectively.

- **Handling Class Imbalance**: Applied **Tomek Links** undersampling method to handle the imbalance in the target variable (`Churn`). Additionally experimented with **SMOTETomek** as an alternative resampling strategy.

- **Feature Scaling**: Applied **StandardScaler** to normalize numerical features, ensuring a balanced and comparable distribution for model input.

By applying this preprocessing pipeline, we prepared a clean and balanced dataset suitable for effective model training and reliable evaluation.

---

##  Modeling Process

The following models were trained and compared using default hyperparameters:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

All models were evaluated using the same training and test sets, split with a 70:30 ratio.

---

## Evaluation

We used the following metrics to evaluate model performance:

- **Accuracy**: Overall correctness of predictions.
- **Precision**: Proportion of positive predictions that are actually churn.
- **Recall**: Ability to detect actual churners.
- **F1-Score**: Harmonic mean of precision and recall, important for imbalanced data.

### Evaluation Results

| **Model**             | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|-----------------------|--------------|---------------|------------|--------------|
| **Gradient Boosting** | 78.42%       | 77.92%        | 78.42%     | 78.14%       |
| SVM                   | 77.89%       | 77.07%        | 77.89%     | 77.39%       |
| Logistic Regression   | 77.75%       | 77.72%        | 77.75%     | 77.73%       |
| Random Forest         | 77.36%       | 76.89%        | 77.36%     | 77.10%       |

### Best Model Selection
The **Gradient Boosting** model is selected as the best-performing model based on its highest test accuracy and F1-score among all evaluated models. Despite its superior performance, further tuning and evaluation may be conducted to optimize the model's effectiveness.

---
## Final Insights & Business Recommendations

- The **Gradient Boosting** model delivered the best performance in predicting customers at risk of churn.  
- Factors such as **Tenure**, **Monthly Charges**, and **Total Charges** had the most significant impact on customer churn decisions.  
- This model can be utilized by companies like DQLab Telco to implement **targeted interventions** and **retention strategies** for customers at risk of churning.

---

###  Key Findings

1. **Tenure is the Dominant Factor**  
   The length of subscription (`tenure`) is the most important predictor. Customers who have stayed longer are typically more satisfied and less likely to churn.

2. **Monthly and Total Charges Are Strong Predictors**  
   Features like `MonthlyCharges` and `TotalCharges` are highly influential. Customers with higher bills may be more sensitive to service issues, while total charges reflect lifetime value.

3. **Other Features Have Minimal Impact**  
   Variables such as `InternetService`, `PaperlessBilling`, and `SeniorCitizen` show lower predictive power. Factors like `gender`, `Partner`, and `StreamingTV` contribute very little to the model’s prediction.

---

###  Practical Implications

- **Retention Strategy**  
  Focus retention efforts on **new or short-tenure customers**, as they are more prone to churn.

- **Customer Segmentation**  
  Identify and cater to **high-paying customers** with special offers, better support, or personalized services to enhance loyalty.

- **Model Simplification**  
  Consider removing **low-impact features** to streamline the model and enable real-time applications without compromising performance.

By leveraging these insights, DQLab Telco can implement **targeted, cost-effective strategies** to proactively reduce churn and improve customer satisfaction.



---

##  Potential Improvements

- **Advanced Hyperparameter Tuning**  
  Explore `GridSearchCV` or `RandomizedSearchCV` to find optimal parameter settings and improve model performance.

- **Feature Engineering**  
  Create interaction features or apply domain-specific transformations to enhance predictive power.

- **Feature Selection & Noise Reduction**  
  Use techniques like Recursive Feature Elimination (RFE) or SHAP values to identify and drop low-impact or redundant features.

- **Addressing Class Imbalance**  
  Experiment with other resampling techniques like **SMOTETomek**, **ADASYN**, or **cost-sensitive algorithms** that directly optimize for minority class (churn).

- **Model Ensemble Techniques**  
  Apply ensemble strategies like **stacking**, **blending**, or **bagging** to combine multiple model strengths and reduce variance or bias.

- **Real-Time Model Deployment**  
  Build an interactive dashboard or API using **Streamlit**, **Flask**, or **FastAPI** to deploy the model for real-time inference.

- **Pipeline Automation**  
  Use tools like **MLflow** or **scikit-learn Pipelines** to streamline data preprocessing, training, and evaluation for reproducibility and scalability.

