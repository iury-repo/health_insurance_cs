# Health Insurance Cross-Sell <img width="55" height="55" alt="image" src="https://github.com/user-attachments/assets/cf069894-1cca-4e28-8df9-ca5eb7f1fb25" align='center' class="center"/>
<img src='https://github.com/iury-repo/health_insurance_cs/blob/main/references/car_insurance.png' alt='health_img' witdh='550' align='center' class="center"/>

**Disclaimer:** This project was inspired by the Kaggle challenge dataset [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction).
All procedures applied and steps made are the standard for real world projects and can be adapted for more complex datasets. 

---

## Business Problem

**Lifebridge Insurance** provides annual health insurance to a consolidated base of 381,109 customers. The company’s CEO wants to convert part of this base to a new product — car insurance. To achieve this, the sales team launched a cross-sell campaign that initially aimed to reach a large number of customers through phone calls, asking whether they would be interested in the new car insurance product.

However, there are several issues with this approach. The calls were made without any customer selection criteria, and as a result, the CEO reported the following about the campaign: Customer Acquisition Cost (CAC) is becoming too high, the calls have a low conversion rate (only 12%), and the sales team has been forced to make decisions “in the dark” in an attempt to improve performance.

Given this scenario, the CEO initiated a data science project to better guide decision-making for the cross-sell campaign, aiming to reduce customer conversion costs and maximize the campaign’s net profit. Additionaly, as the company needs to mitigate part of the damage caused by the poor performance of the cross-sell campaing, the sales team was instructed to reduce the number of phone calls to 20,000. Thus, the project has the following objectives:

 - Reduce the CAC of the cross-sell campaign.

 - Increase the conversion rate across marketing channels.

 - Prioritize customers with a higher probability of conversion.

## Strategy for Solution and Dataset Nature

Last year, when clients subscribe for health insurance, a survey was taken as part of the cross-sell campaign. The survey was made prioritizing relevant vehicle information to the insurance process like number of vehicles, age, driving license, if client vehicle's was damaged at some point in the past, how old are the vehicle, etc... And the dataset that will be used in this project is the result of the information collected in this surveys.

The strategy to propose a solution through machine learn was guided by the reliable CRISP-DM method and the first cycle of development was taken as follow:

**<ins>Business understanding:</ins>** Was fairly discussed in the topic above, so we'll skip this CRISP step for the moment. Feel free to take a look on Business Problem before we proceed.

**<ins>Data understanding:</ins>** 

- **Data Description:** For this project, the dataset was obtained locally via download on the Kaggle page, but as a standard first step, we would extract the data using SQL in a server database or cloud service. The dataset structure is as follows:

| Variable              | Definition                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| id                    | Unique ID for the customer                                                  |
| Gender                | Gender of the customer                                                      |
| Age                   | Age of the customer                                                         |
| Driving_License       | 0: Customer does not have DL, 1: Customer already has DL                   |
| Region_Code           | Unique code for the region of the customer                                  |
| Previously_Insured    | 1: Customer already has Vehicle Insurance, 0: Customer doesn't have it     |
| Vehicle_Age           | Age of the Vehicle                                                          |
| Vehicle_Damage        | 1: Customer got his/her vehicle damaged in the past, 0: Customer didn't    |
| Annual_Premium        | The amount customer needs to pay as premium in the year                    |
| Policy_Sales_Channel  | Anonymized code for the outreach channel (Agents, Mail, Phone, In Person) |
| Vintage               | Number of days the customer has been associated with the company           |
| Response              | 1: Customer is interested, 0: Customer is not interested                   |

 With 381109 rows 0 missing values.

 Looking at the response distribution we can immedeatly notice that the dataset is highly imbalanced, with only 12.3% of the responses being positive. So to deal with this imbalance, we needed to make some decitions when we get to the modeling step.

 <img src="https://raw.githubusercontent.com/iury-repo/health_insurance_cs/main/reports/figures/class_distrib.png" width="300" align='center' class="center"/>


## Useful Insights

## Models Utilized

## Model Performance

All models cumulative gain             | All models lift
:-------------------------:|:-------------------------:
![](https://github.com/iury-repo/health_insurance_cs/blob/main/reports/figures/resampled_cumgain.png)  |  ![](https://github.com/iury-repo/health_insurance_cs/blob/main/reports/figures/resampled_lift.png)


## Business Performance

## Conclusion

---

## Documentation
 

## Notebook Name Convention

```
Notebook name example: 0.01-mni-short-description.ipynb

    0 - Data wrangling / description - often includes cleaning, feature creation and initial diagnostics about the raw dataset. Writes data to data/processed or data/interim
    1 - Data exploration - often just for exploratory work and initial diagnostic about the raw dataset.
    2 - Modeling - training machine learning models
    3 - Visualizations - often writes publication-ready viz to reports
    4 - Publication - Notebooks that get turned directly into reports

* mni - Your initials (My Name Initials); this is helpful for knowing who created the notebook and prevents collisions from people working in the same notebook.
* short-description - A description of what the notebook covers, e.g. (data-cleaning, visualizations, fine-tuning, etc)
```
--------

