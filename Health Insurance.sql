create database Health_insurance;
 show databases;
use health_insurance;

#  View all data

show tables
SELECT * FROM insurance;

# b) Count the number of policyholders
SELECT COUNT(*) AS total_policyholders FROM insurance;

# Total claim amounts by region
SELECT region, SUM(charges) AS total_charges
FROM insurance
GROUP BY region;

#Average claim amount based on smoking status
SELECT smoker, AVG(charges) AS avg_charges
FROM insurance
GROUP BY smoker;

# Identify the region with the highest average claim amount
SELECT region, AVG(charges) AS avg_charges
FROM insurance
GROUP BY region
ORDER BY avg_charges DESC
LIMIT 1;

#Analyze the impact of BMI on claim amounts
SELECT 
    CASE 
        WHEN bmi < 18.5 THEN 'Underweight'
        WHEN bmi BETWEEN 18.5 AND 24.9 THEN 'Normal weight'
        WHEN bmi BETWEEN 25 AND 29.9 THEN 'Overweight'
        ELSE 'Obese'
    END AS bmi_category,
    AVG(charges) AS avg_charges
FROM insurance
GROUP BY bmi_category;

#Region-Specific Analysis
# Average claim amount in each region by smoking status
SELECT region, smoker, AVG(charges) AS avg_charges
FROM insurance
GROUP BY region, smoker;

#Region-specific highest claim amounts
SELECT region, MAX(charges) AS max_charges
FROM insurance
GROUP BY region;

#Predictive Insights Preparation
#v a) Find key attributes influencing higher claims
SELECT 
    age, 
    sex, 
    bmi, 
    children, 
    smoker, 
    charges
FROM insurance
WHERE charges > (SELECT AVG(charges) FROM insurance)
ORDER BY charges DESC;

# Region-based filtering for further modeling
SELECT *
FROM insurance
WHERE region = 'southeast';


# Total claim amounts by region and smoking status
SELECT region, smoker, SUM(charges) AS total_charges
FROM insurance
GROUP BY region, smoker;

#Age distribution of policyholders
SELECT age, COUNT(*) AS policyholder_count
FROM insurance
GROUP BY age
ORDER BY age;
