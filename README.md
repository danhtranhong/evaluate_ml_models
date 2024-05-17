# evaluate_ml_models

## Part 1: Data pre-processing
 Overview description of database and data cleaning(missing values, duplicates,...)The initial phase involves data cleaning, which includes removing irrelevant data, fixing errors, imputing missing values and identifying duplicates. The details are shown in Table 1

### DB1
Instances:  Total: 569 cases || +Malignant: 212 (37%) + Benign: 357 (63%)
Features: 32 columns:
+ Target feature: Class column ( M = malignant, B = benign)
+ Predictor features
(numerical): the standard error, mean, and worst values of ten cell nucleus.
 Missing values: None
Other Accuracy: None

### DB2
Total: 286 cases
+Recurrence events: 85 (30%)
+ No recurrence events : 201 (70%)
10 columns:
+ Target feature: Class column(recurrence-events, no-recurrence-events)
+Predictor features 
(categorical): age, 
menopause,tumor size, 
inv- nodes, node caps, degree of malignancy, breast position, breast 
quadrant and irradiation therapy. 


+node_caps: 8
+breast_quad: 1(denoted by “?”)
All predictor features are categorical
+ Missing values: Replace them with Mode 
+ Encoding 
Categorical features to numerical 
because they are relevant 
to analysis.

### DB3
Total: 961 cases
+ Benign: 516 
+ Malignant: 445 

After cleaning:
Total: 830 cases
+ Benign: 427 (51%)
+ Malignant: 403 (49%)
6 columns:
+Target feature: benign=0 or malignant=1
+ Predictor features:
.BI-RADS assessment(ordinal)  
. Age (integer)
. Shape (nominal)
. Margin(nominal)
. Density (ordinal)

+BI-RADS assessment: 2
+ Age: 5
+ Shape: 31
+Margin: 48
+ Density: 76

Drop missing values.


### DB4
Total: 699 cases
+ Benign: 458
+ Malignant: 241

After cleaning:
Total: 683 cases
+ Benign: 444 (65%)
+ Malignant: 239 (35%)
11 columns:
+ Target feature: benign=2 or malignant=4
+ Predictor features 
(numerical): Uniformity of Cell Shape, Clump Thickness, Uniformity of Cell Size, Mitoses, Bare Nuclei, Bland Chromatin, Normal Nucleoli, Marginal Adhesion, and Single Epithelial Cell Size.


+ Bare Nuclei: 16
None
Drop missing values.

### DB5
Total: 24 cases
+docetaxel-resistant tumors: 14 (58%)
+docetaxel-sensitive tumors: 10 (42%)
9486 features: 
+Target feature:  
docetaxel-resistant tumors and docetaxel-sensitive 
tumors.
+Predictor features 
(numerical): information 
about breast cancer core biopsy from patients

After feature reduction: 8 components
None
Number of predictor features is too large
Reduce dimensionality with Principal Component Analysis (PCA) - 8 components.

### DB6
Total: 198 cases

After cleaning:
Total: 194 cases
+ non-recurrent events: 148 (76%)
+ recurrent events: 46 (24%)
34 features:
+ Target feature: Class: R = recur, N = nonrecur
+ Predictor features 
(Numerical): radius, 
texture, perimeter, area, smoothness,compactness, concavity, concave points, symmetry, fractal 
dimension, tumor size, lymph node status
Lymph node status: 4
None
Drop the missing value

### DB7
Total: 4024 cases
+ Alive: 3408 (85%)
+ Dead: 616 (15%)
16 features:
+ Target feature: Status: Alive and Dead
+ Predictor features:
Nominal (9): Race, Marital Status, T Stage, N Stage, 6th Stage, Grade, A Stage, Estrogen Status, Progesterone Status.
Numerical (5): 'Age', 'Tumor Size', 'Reginol Node Positive', 'Survival Months', 'Regional Node Examined'
None
There are 8 nominal features which are irrelevant to analysis 
Drop the irrelevant nominal features that might be the noises for the model training.

### DB8
Total: 116 cases
+healthy control: 52 (45%)
+ patient: 64 (55%)
10 features:
+ Target feature: Class: 1=Healthy controls, 
2=Patients
+ Predictor feature 
(numerical): Age, BMI, 
Glucose, Insulin, HOMA, 
Leptin, Adiponectin, 
Resistin, MCP-1
None
None



## Diagram
![image](https://github.com/danhtranhong/evaluate_ml_models/assets/143692704/b017a0da-b4d5-45f8-a918-5ec9eb892426)
