# MSBA Capstone – Home Credit Default Risk Prediction

## Business Problem & Project Objective
Many people struggle to get loans due to insufficient or non-existent credit histories and this population is often taken advantage of by untrustworthy lenders. Home Credit, a consumer finance firm provides loans dedicated to the unbanked population. This financial inclusion effort often costs Home Credit as it involves a high risk of loan defaults. However, the business strives to ensure that clients capable of repayment are not rejected from lending. Hence, to be able to take better lending decisions, the firm wants to use historical loan application data to predict whether or not an applicant will be able to repay the loan.

## Solution
The solution to this problem would include the development of supervised predictive classification models since we have historical labelled data and as the outcome of the target variable is binary with values of 0 and 1, indicating that the potential customer will repay the loan on time or will have difficulty repaying loan respectively. This involves Logistic Regression and use of ensemble methods for extracting a list of customers with higher probabilities of loan repayment. The best test AUC and Kaggle score obtained were around 0.76 with the LightXGB model. 

## My Contribution
Though I have a good grip and understanding of the entire project, I was primarily responsible for Under & SMOTE Sampling, Baseline Random Forest, XG Boost and Neural Networks models, ROC-AUC Curves, Test Set Preparation, Feature Importance, XAI & SHAP Plots, Templates for Final Predictions & Model Score Summary. Towards the end I proof Read and made final edits.

## Business Value 
●	Home Credit can use the model to make prediction of future loan applications to confidently make decision of which customers to provide loans to.
●	The SHAP Tree Explainer plot shall empower Home Credit to explain model predictions at a individual application level and will help take informed decision on why a given customer  will be approved/rejected of loan. 
●	This improved decision making will lead to low risk and Non-Performing Assets.
●	The model along with tree explainer plot will lead to a automated, faster, efficient & streamlined loan approval process. 
●	The SHAP value will determine the loan default risk associated with each customer. This shall help Home Credit to tailor credit limits, interest rates based on risk involved. 
●	The model can be used to predict loan repayment behavior for all the current active loan applications to identify customers with greater chances to default. This would allow the firm to take certain proactive measures such as financial counseling, frequently monitoring credit scores and activity in order to prevent defaults in the future. 

## Challenges
Handlining missing values was challenging considering the fact that few of these columns were highly correlated with the target variable. Moreover, tuning and smote sampling of Neural Network was computationally expensive and led to crash of RAM in Google Collab.

## Project Learnings
This project being end to end (from Identifying business problem, EDA, Modeling to deriving business values), it was a good experience to thoroughly delve into each of these stages. I brushed up my knowledge of EDA and Logistic Regression, Random Forest models. In addition, I have had invaluable learnings on researching methodology for feature engineering, GXBM & LGBM and Neural Net models. Towards the end, business value was a basic ROC-AUC curve and feature importance chart in all of my past projects. But this time, I dived into the world of XAI and leveraged SHAP library to explain black box model predictions. This turned out to be very interesting and I look forward to dig deeper into these libraries in my next endeavors. 
