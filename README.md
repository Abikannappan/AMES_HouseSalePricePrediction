# AMES HOUSE SALEPRICE PREDICTION MACHINE LEARNING MULTI LINEAR                MODEL


## Problem Statement
Based on the AMES house saleprice datas, We are here to create a model to evaluate the sale price that could be used by general public who are to buy new home at AMES and also by real estate agents who have to sell or buy new house and they can suggest their customer about the saleprice. Questions that arise involving the project includes, What are the independant variables and how much they affect the saleprice.

## Data Dictionery
Please refer to the following link
http://jse.amstat.org/v19n3/decock/DataDocumentation.txt


## Data Exploration steps
   * Changed column name to lower case and inserting _ for space.
   * Checked for null values and data types of the columns
   * Changed the data types
   * Null value at garage area and garage car is replaced into 0 as some home maynot have a car.
   * Missing value in lot frontage is changed to median as all home will have a lot frontage.
   * Some home may not have basement, So total basement is changed to 0.
   * Statistical datas of each colums are investigated
   * Checked for outliers and data entry errors and corrected the garage_yr_blt from 2207 to 2007
   * Removed the gr_liv_area above 4000 which is a outlier
   * Some home may not have mas_vnr_area, so changed it to 0.
   * Dummie variables created for nominal datas
   * Ordinal variables are changed with numerical values with map() function
   * Some of the house doesn't have a garage and so garage year built is changed to 0
   * Null values at bsmt_full_bath, bsmt_half_bath, bsmt_unf_sf,bsmtfin_sf_1 and bsmtfin_sf_2 without basement is changed to 0
   * I have checked the independant variables correlated with our target saleprice with .corr() and .sort_values() functions.
    
## Model with data
   * Initially I chose my feature variable to construct the model based on correlation. Considered variables which are correlated + or _        0.35 and later dropped and added one variable at a time and checked the model.
   * Created new dataframe df_train with the chosen features
   * The info() about the df_train has mixed datatypes. Except ID and PID all other features are converted into float.
   * Reindex column with .reindex() function
   * Heat map is constructed to check correlation visually
   * gr_liv_area is highly correlated with totrms_abvgrd, fireplace_qu have high correlation with fireplaces. So we can drop fireplaces and      totrms_abvgrd which are less correlated with saleprice compared to fireplace_qu and gr_liv_area. As 1st_flr_sf and total_bsmt_sf are        well correlated a polynomial is created and they are dropped out.
   * Now our features have less correlation with each other but well correlated with saleprice as per multi linear model requirement.
   * Scatter plot is made to check the relation of individual feature with sale price.The features are in a linear relationship either          positive or negtive with our target saleprice as per the requirement for multilinear model.
   * Histogram is constructed to check the distribution of the variables involved in the model.The features are not normally distributed.        So now we are going to train our model with chosen variables.
   * Our X valve is our independant variables stored under features and Y value or our target is saleprice.
   * As per industrial norm the train and test data is split into 75:25%
   ##### LinearRegression()
   - We instantiate LinearRegression() model and try to fit our data
   - -6.25795622e+03,  1.01327160e+04,  1.66390418e+02,  1.47098385e+01, 2.08387069e+03,  2.26261696e+04,  6.74568943e+03,                      4.66776870e+01,7.56099128e+03,  9.34829409e+01,  1.15630213e+03,  1.95247028e+02, 2.18187977e+03,  2.40292195e+01,  5.06668414e+01,        1.34361089e+04, 1.21720276e+04,  1.00007937e+04,  1.54543970e+01 are the beta coefficient for our features                                  'full_bath','mas_vnr_type_None','lot_frontage','wood_deck_sf','heating_qc','neighborhood_NridgHt','bsmt_exposure',
     'mas_vnr_area','fireplaces','year_remod/add','garage_finish','year_built',
     'bsmt_qual','garage_area','gr_liv_area','exter_qual','kitchen_qual','overall_qual','1st_flr_sf + total_bsmt_sf'
   - Increase in 1 sq.ft. of area for 1st_flr_sf + total_bsmt_sf increase the sale price by USD 15.45
   - If the neighbourhood is NridgHt the price increase by USD 22626.16
   - But with increase in full_bath the price show a decrease by -6257.95. While the scatter plot show a positive linear relationship.
   - All other variable follows the trend as per our scatter plot between predictors and target.
   - RMSE is 27121.80
   ##### STANDARDISATION
   Standardisation with StandardScaler()
   ##### RIDGE CROSS VALIDATION  
   - We got optimal alpha with RidgeCV()
   - Then used fit() with optimal alpha
   - prediction done with predict() with optimal alpha
   - Root mean square error is calculated.
   - The RMSE for the ridge is 27087.94 which is lower than the linear regression model 27121.80. So, Ridge model is better.
   ##### LASSO CROSS VALIDATION  
   - We got optimal alpha with LassoCV()
   - Then used fit() with optimal alpha
   - prediction done with predict() with optimal alpha
   - Root mean square error is calculated.
   - The RMSE for the lasso is 27140.55 which is higher than the linear regression model 27121.80 and also the RMSE for the ridge which          is 27587.94.
  ##### ELASTIC NET VALIDATION  
   - We got optimal alpha with ElasticNetCV()
   - Then used fit() with optimal alpha
   - prediction done with predict() with optimal alpha
   - Root mean square error is calculated.
   - The RMSE for the elastic net is 27140.55 which is similar to lasso.
     
#### Repeated   LinearRegression(), Standardisation,  RiddgeCV(), Lasso() and  ElasticNetCV(), fit and predict for following trials
     
    
# Features and Kaggle Score:
## Trial 1
   FEATURES USED:
    'lot_frontage','wood_deck_sf','screen_porch','mas_vnr_type_None','heating_qc','neighborhood_NridgHt','bsmt_exposure',
    'mas_vnr_area','foundation_PConc','full_bath','fireplace_qu','year_remod/add','garage_finish','year_built',
    'bsmt_qual','garage_area','gr_liv_area','exter_qual','kitchen_qual','overall_qual','1st_flr_sf + total_bsmt_sf'
    
   Kaggle RMSE: 33211.91569  
    
## Trial 2
   FEATURES USED:
    'lot_frontage','wood_deck_sf','screen_porch','mas_vnr_type_None','heating_qc','neighborhood_NridgHt','bsmt_exposure',
    'mas_vnr_area','foundation_PConc','full_bath','fireplaces','year_remod/add','garage_finish','year_built',
    'bsmt_qual','garage_area','gr_liv_area','exter_qual','kitchen_qual','overall_qual','1st_flr_sf + total_bsmt_sf'
    
   Kaggle RMSE: 33211.91569  
    
## Trial 3
   FEATURES USED:
    'mas_vnr_type_None','lot_frontage','wood_deck_sf','screen_porch','heating_qc','neighborhood_NridgHt','bsmt_exposure',
    'mas_vnr_area','foundation_PConc','fireplaces','year_remod/add','garage_finish','year_built',
    'bsmt_qual','garage_area','gr_liv_area','exter_qual','kitchen_qual','overall_qual','1st_flr_sf + total_bsmt_sf'
    
   Kaggle RMSE: 33052.04527
    
## Trial 4
   FEATURES USED:
    'mas_vnr_type_None','lot_frontage','wood_deck_sf','screen_porch','heating_qc','neighborhood_NridgHt','bsmt_exposure',
    'mas_vnr_area','foundation_PConc','fireplaces','year_remod/add','garage_finish','year_built',
    'bsmt_qual','garage_area','gr_liv_area','exter_qual','kitchen_qual','overall_qual','1st_flr_sf + total_bsmt_sf'
    
   Kaggle RMSE:  32190.60509    
    
## Trial 5 and Finalised one
   FEATURES USED:
    'mas_vnr_type_None','lot_frontage','wood_deck_sf','screen_porch','heating_qc','neighborhood_NridgHt','bsmt_exposure',
    'mas_vnr_area','fireplaces','year_remod/add','garage_finish','year_built',
    'bsmt_qual','garage_area','gr_liv_area','exter_qual','kitchen_qual','overall_qual','1st_flr_sf + total_bsmt_sf'
    
   Kaggle RMSE:  32152.31164

# RECOMMENDATION
We have created a multi linear model for our sales price prediction. We have used Ridge regression with RMSE of 27087.94 which is lower than the linear regression model 27121.80. The features that highly affect the sales price includes mas_vnr_type_None','lot_frontage','wood_deck_sf','screen_porch','heating_qc','neighborhood_NridgHt','bsmt_exposure',
'mas_vnr_area','fireplaces','year_remod/add','garage_finish','year_built','bsmt_qual','garage_area','gr_liv_area','exter_qual','kitchen_qual','overall_qual','1st_flr_sf + total_bsmt_sf'. Most of them show a positive relationship, when there is a unit increase in independant variable there is a optimal_ridge.coef_ (Value from the following table) times positive increase in the salesprice. While there is a decrease in sales price with increase in full bath. This model works well for sale price until 500000 with Root mean square error of about USD 27087.94 for house at AMES. The predicted saleprice have linear relationship, The Residual histogram follow a normal distribution and the for Residual Vs Test Sale price the values are spread around zero line as per our requirement for linear model, But the plot have one outlier which shows it is only trained until sales of USD 500000.    