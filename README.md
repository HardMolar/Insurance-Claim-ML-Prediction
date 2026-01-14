# AINOW-INCUBATOR-HUB-INSURANCE-CLAIM-CAPSTONE-PROJECT
This is a machine learning project to predict building claims by an Insurance company 
# PALMORA GROUP HR ANALYSIS.

## 📌 Case Study Overview
Palmora Group Manufacturing Company is a big manufacturing hub in Nigeria faced with allegations of gender inequality across its three operating regions. Media criticism and internal leadership concern have prompted an urgent HR data review to identify, assess, and recommend actions for equity, especially regarding pay gaps and performance-based bonus allocation.
As the HR Analytics Consultant, I’ve been tasked with performing a comprehensive analysis to uncover critical gender-based insights and salary disparities within departments and regions.

## 🎯 Objectives
  * Analyze gender distribution by department and region
  * Identify disparities in performance ratings based on gender
  * Evaluate the company’s salary structure for gender pay gaps
  * Assess compliance with salary legislation (minimum of $90,000)
  * Visualize salary band distributions by $10,000 ranges
  * Allocate bonuses based on performance ratings
  * Provide total payout insights by region and overall
    
## 🗃️ Data Sources
  * CSV file of the companies data which include employee details: Names, Gender, Departments, Location and Performance rating.
  * Excel file that contains departments and general ratinngs of each department, which is to be used to calculate bonus for individual staffs based on work rating.

## 🧰 Tools and Technology
  * Microsoft Power BI: For data exploriation
  * DAX: Bonus Calculatins, Conditional logic

# 🧬 Project Structure

## 🚧 Data Exploratory Analysis
### 🧹 Data Cleaning
Before analysis, the following data preparation steps were required:
  * Missing Gender:Assign a generic gender status for employees who did not disclose their gender.
  * Missing Salary:Exclude employees with no salary (likely no longer with the company).
  * Missing Department:Remove employees tagged with "NULL" departments.

## 📌 Case Scenario
  * Determine the gender distribution by region and department
  * Show insights based on gender.
  * Analyse the company's salary structure so as to identify the salary pay.   gap as regards departments and region, which management should focus on.
  * Minimum salary compliance (>=90,000).
    ** Show pay distribution table of employees grouped by a band of $10,000.
    ** Visualization of the salary band by region.
  * Bonus amount to be paid to individual employee.
  * Total amount(Salary+Bonus) to be paid to individual employee.
  * Total amount to be paid per region and worldwide

## 🧠 Insights Generated
1. Gender Distribution by region and department
Visualized by department and region using bar (stacked column chat) for gender distribution by department and donut charts for overall gender distribution by region.

  **Gender distribution across all region**
  Overall Staffs: 946
  Female: 441
  Male: 465
  Neutral: 40
  
  **Staff distribution across all region**
  
  Lagos: 250(26.43%)
  Abuja: 335(35.41%)
  Kaduna: 361(38.16%)
  
  **Gender distribution across all departments**
    
| Department                  | Neutral | Male   | Female  |
|-----------------------------|---------|--------|---------|
| Product Management          | 1       | 47     | 41      |
| Legal                       | 5       | 49     | 34      |
| Human Resource              | 3       | 38     | 41      |
| Services                    | 3       | 37     | 42      |
| Business Development        | 3       | 37     | 41      |
| Support                     | 4       | 42     | 35      |
| Engineering                 | 6       | 36     | 38      |
| Sales                       | 4       | 40     | 36      | 
| Training                    | 3       | 38     | 36      |
| Research and Develpment     | 5       | 31     | 38      |
| Accounting                  | 2       | 37     | 28      |
| Marketing                   | 1       | 33     | 31      |

<img width="794" height="445" alt="HD1" src="https://github.com/user-attachments/assets/1958c2bb-6d8b-48a7-9ae9-ac7feea15dae" />

2. Ratings based on Gender
Clustered chattered chat shows performance trends by gender and Line chat to show rating comparison by gender
  
| Rating                      | Neutral | Male   | Female  |
|-----------------------------|---------|--------|---------|
| Average  		                 | 18      | 212    | 190     |
| Good                        | 9       | 82     | 89      |
| Poor             	          | 5       | 70     | 58      |
| Very Good                   | 3       | 37     | 42      |
| Not Rated        	      `   | 2       | 34     | 35      |
| Very Poor                   | 3       | 31     | 20      |

![BI_002](https://github.com/user-attachments/assets/d2bd0969-5578-4c42-be82-6fea9768b32a)

3. Salary Structure Analysis across all departments and region
* Detected gender pay gaps across certain departments/regions
* Highlighted departments and regions for leadership focus
  
**Overall departmental payment per gender**

| Gender                      | High Pay              | Low Pay        |
|-----------------------------|-----------------------|----------------|
| Neutral  		                 | Marketing             | Human Resources|
| Male                        | Bisuness Development  | Engineering    |
| Female             	        | Marketing             | Human Resources|

**Average Salary Per gender**
Neutral: $78,367.50
Male: $74,789.53
Female: $72,135.69

```dax
Average Salary = AVERAGE('Palmoria Group emp-data'[Salary])
```

**Regional payment gap per gender**
**Abuja**
| Gender                      | Departmets that pay high                                       |        
|-----------------------------|----------------------------------------------------------------|
| Neutral  		                 | Marketing,Accounting, Human Resources, Research and Development|
| Male                        | Marketing                                                      |
| Female             	        | Marketing                                                      |

**Lagos**
| Gender                      | Departmets that pay high          |        
|-----------------------------|-----------------------------------|
| Neutral  		                 | Accounting, Legal                 |
| Male                        | Accounting, Training              |
| Female             	        | Accounting                        |

**Kaduna**
| Gender                      | Departmets that pay high          |        
|-----------------------------|-----------------------------------|
| Neutral  		                 | Training, Sales                   |
| Male                        | Accounting, Training              |
| Female             	        | Engineering                       |

![BI_003](https://github.com/user-attachments/assets/ead7b2b9-a60c-4db9-b784-42cdb9a8a778)


4a. Minimum Salary Compliance
Analyzed employees earning below the $90,000 threshold, which = **654(69.13%)**
This is indeed a red flag 🚩 because more than 50% of the total employee are being paid below the recomended minimum salary threshhold.

```dax
Employees Below Minimum Salary = COUNTX(FILTER('Palmoria Group emp-data', 'Palmoria Group emp-data'[Salary] < 90000), 'Palmoria Group emp-data'[Name])
```

![BI_004a](https://github.com/user-attachments/assets/332fc32a-9866-4403-ab99-b872e2d2abb8)


4b. Salary Bands Distribution across all region

| Salary Band ($)             | Employee Count    |        
|-----------------------------|-------------------|
| >100,000 		                 | 202               |
| 90,001 - 100,000            | 90                |
| 80,001 - 90,000             | 108               |
| 70,001 - 80,000             | 117               |
| 60,001 - 70,000             | 99                |
| 50,001 - 60,000             | 96                |
| 40,001 - 50,000             | 105               |
| 30,001 - 40,000             | 101               |
| 20,001 - 30,000             | 28                |
| **Total**                   | **946**           |

![BI_004b](https://github.com/user-attachments/assets/f848e481-e2de-4a89-9d84-b7aecf13ef95)


4c. Visualized by gender and region for comparative analysis

5a. Bonus Allocation
Calculated bonus per employee based on performance criteria
Computed:
Bonus amount per employee

```dax
Bonus Amount = SUMX('Palmoria Group emp-data', 'Palmoria Group emp-data'[Salary] * LOOKUPVALUE('Bonus Rules'[Value], 'Bonus Rules'[Department Rating], 'Palmoria Group emp-data'[Department Rating]))
```

5b. Total amount to be paid

```dax
Total Amount to be Paid = SUMX('Palmoria Group emp-data', 'Palmoria Group emp-data'[Salary] + [Bonus Amount])
```

![BI_005a:b](https://github.com/user-attachments/assets/aa851520-ae4d-4308-a848-151b8d0908ca)

5c. Regional and company-wide bonus payout totals

**Regional Payment**
kaduna = $27.48 million
Abuja = $24.93 million
Lagos = $19.53 million

**Company-Wide Payment**
$71.94 million

![BI_005c](https://github.com/user-attachments/assets/6e02d9dd-db52-43c2-b1d2-7e1f36d732e5)


## ✅ Recomendations
1. Address gender pay gaps, especially in flagged departments/region
2. Improve transparency in performance evaluation metrics
3. Consider pay restructuring to meet legal compliance
4. Implement regular reviews of pay and performance equity
5. Develop HR policies encouraging full data disclosure 



