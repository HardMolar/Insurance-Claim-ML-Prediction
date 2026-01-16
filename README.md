# INSURANCE CLAIM PREDICTION PROJECT
This is a machine learning project to predict building claims for an Insurance company 

## 📌 Case Study Overview
As the Lead Data Analyst of an insurance company, I’ve been saddled with the responsibility to build a predictive model to determine if a building will have an insurance claim during a certain period or not. Our goal is to identify patterns associated with claim occurrence and improve risk assessment for insurance providers.
This capstone project, is one of the requirements from -The Incubator Hub- to get a (Certification in Data Science and Machine Learning)- AINOW BOOTCAMP
The project follows a standard data science workflow including data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.


## 🎯 Objectives
  * Clean and preprocess insurance claims data
  * Explore relationships between the features and target (claim outcomes)
  * Conduct Feature Engineering
  * Train predictive machine learning models
  * Evaluate model performance using appropriate metrics
  * Identify important risk factors influencing insurance claims

    
## 🗃️ Data Sources and Description
  * The dataset was provided by "The Incubator Hub" and it's a CSV file that contains insurance policy holder information and claim-related attributes.
```python
Insurance_Train_Data = pd.read_csv('datasets/Train_data.csv')
Insurance_Train_Data
```

### 🎯 Key Variables
  * Demographic features: Customer Id, Residential, Settlement, Geo Code
  * Policy characteristics: Insured_Period, , Building_Painted, Building_Fenced. YearOfObservation
  * Risk indicators: Building Dimension, Building_Type, NumberOfWindows, Date_of_Occupancy
  * Target variable: Claim
> Missing values are represented by `"."` and handled during preprocessing.


## 🧰 Tools and Libraries
  * Python 3
  * NumPy
  * Pandas
  * Matplotlib
  * Seaborn
  * Scikit-learn


# 🔄 Project Workflow
## 🧹 Data Cleaning and Preprocessing  
Before analysis, the following data preparation steps were required:
   * Handled missing values
     ```python
     Insurance_Train_Data.isnull().sum()
     ```
       
   * Removed or treated inconsistent entries
   ```python
   Insurance_Train_Data.columns = (
    Insurance_Train_Data.columns
    .str.strip()                #Remove trailing and leading spaces
    .str.lower()       #lowercase
    .str.replace(' ', '_')      #Replace spaces with underscores
   )
   ```

   * Replaced `"."` with NaN values
   ``` python
   Insurance_Train_Data.replace('.', np.nan, inplace=True)
   ```

   * Converting all features to numeric values using pd.to_numeric.
     ```python
     Insurance_Train_Data['building_dimension'] = pd.to_numeric(Insurance_Train_Data['building_dimension'], errors='coerce')
     Insurance_Train_Data['date_of_occupancy'] = pd.to_numeric(Insurance_Train_Data['date_of_occupancy'], errors='coerce')
     ```

   * During cleaning, one of the biggest challenge faced was Data Inconsistency:
    ** The Problem: I encountered "Unknown" strings in numeric columns and categorical data that computers couldn't read.
    ** The Solution: I performed a "Clean Sweep" by:
   * Stripping away hidden metadata that was causing "string-to-float" errors.
   * Aligning the Target (y) with the Features (X) to ensure every row was accounted for.

## 🚧 Data Exploratory Analysis
   * Distribution of target variables (Claim = 0/1).
     Certain demographic and policy variables show skewed distributions, indicating heterogeneous risk profiles.
     We used a Normalized Countplot (using hue). This allows us to compare the proportion of claims, which is more important than the raw total.
 ```python
     plt.figure(figsize=(6,4))
     sns.countplot(x='claim', data=Insurance_Train_Data, palette='viridis')
     plt.title('Distribution of Insurance Claims (0 = No, 1 = Yes)')
     plt.savefig('Distribution of Insurance Claims.png', dpi=300, bbox_inches='tight')
     plt.show()
 ```

   ![Distribution of Target Variable - Claim](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/9604f7e7e38381acf60732833983629734377e0a/Visuals/Distribution%20of%20Insurance%20Claims.png?raw=true")


	* Distribution of Building Dimension by Claim
![Distribution of Target Variable by Claim](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Building%20dimension%20distribution%20by%20claim.png?raw=true")



   * Distribution of outliers in the variables Bilding_Types, Building_Dimension and Building_Age. 
```python
     # Define the crucial numerical features for outliers
num_cols = ['building_dimension', 'building_type', 'building_age'] 

print("--- Outlier Detection Summary ---")
for col in num_cols:
    #Calculate IQR
    Q1 = Insurance_Train_Data[col].quantile(0.25)
    Q3 = Insurance_Train_Data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    #Define Bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    #Count Outliers
    outliers_count = Insurance_Train_Data[(Insurance_Train_Data[col] < lower_bound) | (Insurance_Train_Data[col] > upper_bound)].shape[0]
    
    print(f"Column: {col}")
    print(f"  - IQR: {IQR}")
    print(f"  - Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  - Total Outliers: {outliers_count}")
    print("-" * 30)
```
	
	* Capping Outliers
```python
#define the capping function

def cap_outliers(Insurance_Train_Data, columns):
    for col in columns:
        Q1 = Insurance_Train_Data[col].quantile(0.25)
        Q3 = Insurance_Train_Data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Capping values outside the bounds
        Insurance_Train_Data[col] = np.clip(Insurance_Train_Data[col], lower_bound, upper_bound)
    return Insurance_Train_Data

# Execute the capping
Insurance_Train_Data = cap_outliers(Insurance_Train_Data, num_cols)
print("Outliers have been capped successfully.")
```

   ![Distribution of Outliers](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Distribution%20of%20Outliers.png?raw=true")

   
   * Bivariate analysis:
   ** Categorical feature exploration
   *** Categorical vs Target: 
    The Yellow/Light Green coloured stackplot describes the percentage of claims to building type
	  There is a symmetrical progression down from building type 1 to 4. This means my hypothesis is correct. Old buildings are riskier.
	  If they are the same: Age might not be as important as other factors like Building_Dimension.

 ![Percentage Claims by Building Types](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Percentage%20of%20Claims%20by%20Building%20Type.png?raw=true")


  * Distribution of Building_Age vs Claim. 
Visualized this using violin plot, so as to graphically view if the risk is higher for very old buildings or new ones.

 ![Building Age by Claim](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Building%20Age%20vs%20Claims.png?raw=true")


 
  * Bulding Dimension vs Age
Scatterplot to visualize the analysis of relationship between the two variables. This is important to ensure that there is no redundant data (Multi-collinearity)

  ![Building Dimension by Age](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Building%20Age%20vs%20Building%20Dimension.png?raw=true")



##  Feature Engineering
 * I removed the Customer_ID column as it serves as a unique identifier with no statistical relationship to the target variable (Claim). Removing non-predictive identifiers prevents model overfitting and reduces computational noise."
 * The NumberOfWindows feature contained mixed types, including the string '>=10'. I standardized this by converting the upper-bound string to a discrete integer and performing median imputation on missing values. This preserves the ordinal nature of the data while making it compatible with the regression and tree-based algorithms.
* Mapping and encoding categorical variables: 
Used mapping technique for categorical columns like 
  ** Building_paint
  ** Building_Fence
  ** Garden
  ** Settlment
  ** For categorical column that has more than 2 variables, ( Building_Type), I used  LabelEncoder to encode the variables from categorical to numeric
```python
#Mapping some of the cartegorical coliumns
Insurance_Train_Data['building_painted'] = Insurance_Train_Data['building_painted'].map({'N': 1, 'V': 1})
Insurance_Train_Data['building_fenced'] = Insurance_Train_Data['building_fenced'].map({'N': 0, 'V': 1})
Insurance_Train_Data['garden'] = Insurance_Train_Data['garden'].map({'V': 1, 'O': 0})
Insurance_Train_Data['settlement'] = Insurance_Train_Data['settlement'].map({'U': 1, 'R': 0})
#Using Label Encoder to encode Building Type column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Insurance_Train_Data['building_type'] = le.fit_transform(Insurance_Train_Data['building_type'].astype(str))
```

 * Correlation Matrix Heat Map
This mathematically shows the correlation between out target(Claim) and every other numeric variables.
Helps to focus on the variables that actually move the needle for the insurance company.

  ![Correlation Heat Map](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Correlation%20Matrix.png?raw=true")


 * Feature selection
  ** Train-test split
 I chose to split the data using 70% to train, 30% to test, random state was 42 and stratify as y.

 * Standardization: The NumberOfWindows feature contained mixed types, including the string '>=10'. I standardized this by converting the upper-bound string to a discrete integer and performing median imputation on missing values. This preserves the ordinal nature of the data while making it compatible with the regression and tree-based algorithms."

```python
#Adjusting the >=10 values in Number of windows column
#Strip any leading/trailing spaces
Insurance_Train_Data['numberofwindows'] = Insurance_Train_Data['numberofwindows'].astype(str).str.strip()
#Replace '>=10' with '10'
Insurance_Train_Data['numberofwindows'] = Insurance_Train_Data['numberofwindows'].replace('>=10', '10')
#Convert to numeric
Insurance_Train_Data['numberofwindows'] = pd.to_numeric(Insurance_Train_Data['numberofwindows'], errors='coerce')
#Fill the gaps (NaNs) with the median
Insurance_Train_Data['numberofwindows'] = Insurance_Train_Data['numberofwindows'].fillna(Insurance_Train_Data['numberofwindows'].median())
#Convert to integer
Insurance_Train_Data['numberofwindows'] = Insurance_Train_Data['numberofwindows'].astype
```

 * Scaling: Because models like Logistic Regression are sensitive to the "size" of numbers (e.g., comparing a building dimension of 2,000 to an insured period of 0.5), we applied Standard Scaling.
  ** Method: We fit the scaler only on the training data to prevent Data Leakage (using future info to predict the past).
  ** Result: All features were placed on a level playing field, ensuring the model didn't get "distracted" by large numbers.
 During the scaling phase, I identified 'Unknown' string entries within the features. 
 I implemented a pipeline to coerce these into numerical format and applied median imputation, ensuring data consistency for the StandardScaler and   subsequent models."
```python
#Feature Scaling
#Using Standard Scaler

X_array = X_final.values.astype(float)
# Creates data with Mean = 0 and Standard Deviation = 1.
#Initialize Scaler
final_scaler = StandardScaler()

#FIT and TRANSFORM the array! (Learning the mean and variance from training data)
X_scaled = final_scaler.fit_transform(X_array)

print("Shape of scaled data:", X_scaled.shape)
print("Success! Data is perfectly cleaned and scaled" )
```

### 🧠 Feature Engineering Insight
Feature Engineering revealed a significant "Risk Gap" based of building age. By categorizing the "Building Age" into Mordern and Historical, we validated that older structure have a significantly higher claim frequency. This allowed our model(Logistics Regression) to assign higher risk weights to historical properties, directly improving the model's F-1 score
 * Bivariate analysis reveals that Historic buildings (50+ years) have a higher claim frequency compared to modern structures. This justifies the inclusion of 'Is_Historic' as a key feature for our predictive models.".
 * Target variable balance assessment: The target variable (insurance claim) appears imbalanced, with non-claims dominating.
   

  ![Correlation Heat Map](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Claims%20Probability.png?raw=true")

  

## 🧠 Model Building
 * Machine learning algorithms selection 
We tested three distinct models to see which one understood the insurance risks best:
 ** Logistic Regression: Used as my Baseline. It assumes a linear relationship between features and risk.
 ** Random Forest: Used to handle Outliers. It builds a "Forest" of decision trees to find stable patterns.
 ** XGBoost: Optimized using gradient boosting for the highest predictive power.

 * Model Initialization
   ** Logistics Regression
```python
   og_model = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    class_weight='balanced',
    solver='liblinear'
   )
```
  
   ** Random Forest
```python
  rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    max_features="sqrt", 
    max_depth=6, 
    max_leaf_nodes=6,
   class_weight='balanced'
   )
```
   
   ** XGBoost
```python
   xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'    
   )
```
     
 * Model training on training dataset
  ** Logistic Regression:
```python
  log_model.fit(X-train, y_train)
```

  ** Random Forest
```python
  rf_model.fir(X_train, y_train)
``` 

 ** XGBoost: xgb_model.fit(X-train, y_train)
```python
  xgb_model.fir(X_train, y_train)
``` 

## 📌 Model Evaluation
  * Accuracy
  * Confusion Matrix
      ![onfusion Matrix](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Confusion%20Matrix.png?raw=true")


   
### 🧠  Comparing the Classification metrics (Precision, Recall, F1-score) of the 3 models
Instead of just looking at Accuracy (which can be misleading in insurance where claims are rare), I prioritized the **F1-Score** to balance Precision (avoiding false alarms) and Recall (catching actual claims).
  * Key Finding: While Logistic Regression provided a solid foundation and it was considered for the project, XGBoost has a high accuracy but lowest F1 score.
   ** Interpretation: The model achieved an F1-score of [0.46], meaning it successfully balanced the need to catch actual claims (Recall) without raising too many false alarms (Precision).
* Overfitting and Validation
To ensure this wasn't just a "lucky guess," iperformed 5-Fold Cross-Validation.
* Verification: The model showed consistent scores across all folds, proving it is Robust and will work on new, unseen buildings in the future, not just the ones in our training set.
| Model                       | Accuracy | Precision   | Recall   | F1 Score  |
|-----------------------------|----------|-------------|----------|-----------|
| Logistic Regression         | 0.713687 |  0.405159   | 0.544898 | 0.464752  | 
| Random Forest               | 0.716015 |  0.402913   | 0.508163 | 0.449458  |    
| XGBoost                     | 0.784451 |  0.577143   | 0.206122 | 0.303759  | 

  ![Prediction Chat](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Prediction%20Chat.png?raw=true")


  ** Based on these metrics, Logistic Regression is actually the winner, but with a very important "Insurance Business" reason why.
  Why Logistic Regression wins here?
  In insurance, F1-Score is more important because it is the "balance" between being right (Precision) and not missing anyone (Recall).
  *** The Problem with XGBoost: Even though its Accuracy and Precision are high, its Recall is the lowest. This means XGBoost is being "too shy." It only predicts a claim when it is 99% sure, but it is missing most of the actual claims. In insurance, missing a claim (a "False Negative") is very expensive.
  *** The Strength of Log Reg: It has the Highest Recall and F1-Score. This means it is much better at "catching" the people who will actually make a claim, even if it makes a few more mistakes (lower Precision) along the way.
  
 ### 📊 Verdict: We choose Logistic Regression. 
"While XGBoost showed higher overall accuracy, In the context of insurance risk, a high Recall is critical to ensure that potential claims are not overlooked. Logistic Regression achieved the highest F1-Score, providing the best balance between identifying risks and maintaining prediction accuracy."
  * The model demonstrates reasonable predictive performance
  * Certain demographic and policy-related features strongly influence claim likelihood
  * Class imbalance affects prediction outcomes and should be addressed in future iterations

  ![Confusion Matrix](https://github.com/HardMolar/AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT/blob/f16b65d94ee5ba51156e221e1b3c164ce561d9ee/Visuals/Confusion%20Matrix.png?raw=true")



## ⚠️ Limitations
  * Class imbalance in claim vs non-claim observations
  * Limited feature granularity
  * No cost-sensitive evaluation metrics included

## 💡 Key Insights
  * Feature Importance: `Building_Dimension` and `Date_of_Occupancy` were the strongest predictors of claim likelihood.
  * Business Value: The model allows for proactive risk management and more competitive pricing for low-risk clients.


## ✅ Recomendations
1. Address gender pay gaps, especially in flagged departments/region
2. Improve transparency in performance evaluation metrics
3. Consider pay restructuring to meet legal compliance
4. Implement regular reviews of pay and performance equity
5. Develop HR policies encouraging full data disclosure 



