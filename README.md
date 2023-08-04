# Well location selection

The aim of this project is to determine the most profitable region for a new oil well. We were provided with oil samples from three regions, and for each region, we have data on 10,000 fields. The quality of oil and the volume of its reserves were measured for each field. A machine learning model must be built to help determine the region where mining will bring the most significant profit. The analysis of possible gains and risks should be done using the Bootstrap technique.

## Steps to Choose a Location
The following steps were taken to select the best location for the new oil well:

1 .In the selected region, a deposit is searched, and for each, the values of the features are determined.
2. Building a model and assessing the volume of reserves.
3. Search for the deposit with the highest value score. The number of fields depends on the company's budget and the cost of developing one well.
4. The profit is equal to the total profit of the selected deposits.

## Libraries Used
The following libraries were used in this project:

- Pandas
- Numpy
- Matplotlib
- SciPy
- Scikit-learn
- Itertools


## Datasets
We were provided with historical data for three regions where 100,000 wells were found. Each well was assigned an id number. For each of the wells, calculations were made of the parameters that affect the quality and quantity of oil in it. The last product column shows the actual amount of inventory.

## Data Preparation
The provided datasets were analyzed, and no complete duplicates were found between all regions, which means that the areas do not intersect with each other, and the samples are not pairwise interconnected. No gaps were found, and the data types are suitable for training and validating our model.

Judging by the values of the target attribute product, we are faced with the task of regression. The Linear Regression model will be suitable for this task.

## Preparing for Profit Calculation
When exploring a region, 500 points are explored, from which the best 200 will be selected for development. The company allocates 10 billion rubles for the development of 200 outlets. It is necessary to determine the most promising region for investment, for which wells have data f0, f1, and f2. It is essential to define the areas so that the probability of loss (risk) does not exceed 2.5%.

One barrel of raw materials brings 450 rubles of income at current prices.

## Profit and Risk Calculation Plan
The profit and risk calculation plan is as follows:

1. 1000 times with incredible randomness, we will take 500 wells from the region's predicted target features.
2. Find relevant actual target features.
3. Calculate profit: 
- Sort our predictions and take 200 indices from the top 200. 
- Use these indices to find the corresponding actual values. 
- Sum them up, multiply them by the cost of a barrel, and subtract the budget from this.
4. Write all 1000 entries with profit calculation and calculate the confidence interval, average, and risks.
5. Do all these operations for three regions.
6. Select suitable regions. If the risk is higher than 2.5%, the region is unsuitable.

## Function Created for Profit and Risk Calculation
The following function was created to calculate profit and risk:
```pythonn
def profit_n_risk(predict, target, color):
    revenue = []
    
    for _ in tqdm(range(1000), colour = colour, desc = 'Loading'):
        predict_sample = predict.sample(oil_points_max, replace=True, random_state=state)
        target_sample = target[predict_sample.index]
        
        revenue.append(profit_count(predict_sample, target_sample))   
    revenue = pd.Series(revenue)
    
    # Start of confidence interval
    lower = revenue.quantile(0.025)
    
    # End of confidence interval
    higher = revenue.quantile(0.975)
    
    # Average revenue per well
    μ_revenue = int(round(sum(revenue) / len(revenue),0))
    
    # What is the percentage of negative average revenue (unprofitable wells)
    risk = st.percentileofscore(revenue, 0)
    
    # We leave only those regions in which the probability of losses is less than 2.5% and give instructions
    to_drill = 'To drill'
    not_to_drill = 'Not to Drill'
    
    
    if risk < 2.5:
        return ((int(lower), int(higher)), revenue, μ_revenue, risk, to_drill)
    else: 
        return ((int(lower), int(higher)), revenue, μ_revenue, 'Higher than 2,5%', not_to_drill)
```
We then run the function for all three regions, calculate the confidence interval, average, and risks.

The results of our analysis will help determine the most promising region for investment, for which wells have data f0, f1, and f2. It is essential to define the areas so that the probability of loss (risk) does not exceed 2.5%.

## Conclusion:

We have analyzed the data and built a Linear Regression model that helps us predict the amount of oil in each well. We then used this model to calculate the profit and risks for each region and determine the most promising region for investment.

Using the initial data, we trained a model that estimated the volume of oil reserves according to the characteristics of the field. Based on this data, we were able to forecast the potential profit from 200 of the top 25,000 wells in each region.
    
The situation for all three regions is as follows:

|Region|Average profit|Probability of loss|
|-----|-----|----|
| Region 1 | ≈ 420 million rubles | Above 2.5% |
| Region 2 | ≈ 520 million rubles | Below 2.5% |
| Region 3 | ≈ 420 million rubles | Above 2.5% |

It is best to choose the second region for drilling. This is also clearly demonstrated by the confidence interval values for all areas.
    
|Region|Start|End|
|-----|-----|----|
| Region 1 | ≈ -100 million | ≈ 950 million |
| Region 2 | ≈ 100 million | ≈ 950 million |
| Region 3 | ≈ -110 million | ≈ 990 million |

Negative revenue is so unlikely for the second region that it does not fall within the confidence interval. If you believe the 95% confidence interval, then the probability of losses is zero. At the same time, the potential profit value is approximately the same as in the other areas and amounts to a little more than 900 million rubles.
    
The only thing that can affect the success of choosing wells in the second region for the current calculation is a drop in the price per barrel of oil.

Overall, this project shows the importance of data analysis and machine learning in decision-making processes for businesses. By leveraging data-driven insights, companies can make informed decisions that can lead to higher profits and better outcomes.
