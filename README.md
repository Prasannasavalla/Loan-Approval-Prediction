### **Loan Approval Prediction System**



#### Project Overview



The Loan Approval Prediction System is a machine learning project designed to predict whether a loan application will be Approved or Rejected based on applicant details such as income, education, employment status, assets, credit score (CIBIL), and loan-related factors.



###### This project demonstrates a complete data science pipeline, including:



* Data preprocessing
* Feature engineering
* Exploratory Data Analysis (EDA)
* Model building
* Model evaluation



##### Objective



To build and compare machine learning models that accurately predict loan approval outcomes, helping financial institutions make data-driven and risk-aware lending decisions.



##### 🗂️ Dataset Description



The dataset contains information about loan applicants, including:



loan\_id ,no\_of\_dependents ,education ,self\_employed ,income\_annum ,loan\_amount ,loan\_tenure ,cibil\_score ,bank\_assets\_value ,luxury\_asset\_value ,residential\_asset\_value ,commercial\_asset\_value ,loan\_status (Target Variable)



📁 Dataset file: loan\_approval\_dataset.csv



##### ⚙️ Technologies Used



###### Programming Language: Python



Libraries:



* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn



##### Project Workflow



###### 1\. Data Loading \& Cleaning



* Removed unnecessary columns (loan\_id)
* Checked for missing values and data types



###### 2\. Feature Engineering



* Created meaningful features to improve model performance:
* Movable Assets = Bank Assets + Luxury Assets
* Immovable Assets = Residential Assets + Commercial Assets
* Original asset columns were removed after feature creation.



###### 3\. Exploratory Data Analysis (EDA)



Visual insights were generated using:



* Count plots for dependents and employment status
* Box and violin plots for income distribution
* Histograms for CIBIL score
* Scatter plots for assets vs loan amount
* Asset distribution comparison by loan status
* These visualizations helped understand patterns influencing loan approval.



###### 4\. Encoding Categorical Variables



Converted categorical values into numerical form:



* Education: Graduate → 1, Not Graduate → 0
* Self Employed: Yes → 1, No → 0
* Loan Status: Approved → 1, Rejected → 0



###### 5\. Correlation Analysis



A heatmap was used to analyze relationships between features and identify important predictors of loan approval.



###### 6\. Model Building



Two classification models were implemented:



* Decision Tree Classifier
* Random Forest Classifier



The dataset was split into:



70% Training

30% Testing

(using stratified sampling)



###### 7\. Model Evaluation



Models were evaluated using:



**Accuracy score**

**Classification report (Precision, Recall, F1-Score)**

**Confusion matrix visualization**



##### 📊 Result:

Random Forest performed better than Decision Tree, showing higher accuracy and balanced performance.



##### 📈 Results \& Insights



* Higher CIBIL score, income, and asset value strongly increase approval chances.
* Random Forest provides more stable and accurate predictions.
* Feature engineering significantly improves model effectiveness.



##### 📂 Project Structure

Loan-Approval-Prediction/

│

├── loan\_approval\_dataset.csv

├── main.py

├── README.md



##### How to Run the Project?



###### Clone the repository



Install required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn



###### Run the script:

python main.py



##### Future Enhancements



* Add Logistic Regression and XGBoost models
* Perform hyperparameter tuning
* Deploy the model using Flask or Streamlit
* Add real-time user input prediction interface



##### Author



Prasanna Savalla

Machine Learning \& Data Science Enthusiast

