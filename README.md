# AMES HOUSE SALEPRICE PREDICTION MACHINE LEARNING MULTI LINEAR                MODEL


## PROBLEM STATEMENT

Based on the AMES housing sales price data, we have to create a multi linear regression model to predict the sales price. The prediction can help the general public and real estate agents, who are looking to buy or sell houses. This model will help the customer in predicting the saleprice. The model is evaluated using RMSE, Kaggle RMSE and Adjusted R squared.

## DATA DICTIONERY

Please refer to the following link
http://jse.amstat.org/v19n3/decock/DataDocumentation.txt


## DATA EXPLORATION AND CLEANING

   * Concatinating test and train set together to impute null values and to do one hot coding
   * Changed column name to lower case and inserting _ for space.
   * Checked for null values and data types of the columns
   * Changed the data types
   * Null value at garage area and garage car is replaced into 0 as some home maynot have a car.
   * Missing value in lot frontage is changed to median as all home will have a lot frontage.
   * Some home may not have basement,mas_vnr_area and garage So total basement details,mas_vnr_area and garage year built are changed to 0.
   * Statistical datas of each columns are investigated
   * Checked for outliers and data entry errors and corrected the garage_yr_blt from 2207 to 2007
   * Removed the gr_liv_area above 4000 which is a outlier
   * Dummy variables created for nominal datas using one hot encoding
   * Ordinal variables are changed into numerical values with map() function
   * I have checked the independant variables correlated with our target saleprice with .corr() and .sort_values() functions.
   * Test and Train datas are splitted.
    
## EXPLORATORY DATA ANALYSIS AND FEATURE SELECTION

   * RFECV under sklearn.feature_selection features are shortlisted which includes
      ['1st_flr_sf', '2nd_flr_sf', 'bsmt_unf_sf', 'bsmtfin_sf_1',
       'bsmtfin_sf_2', 'exter_qual', 'gr_liv_area', 'low_qual_fin_sf',
       'total_bsmt_sf', 'ms_subclass_150', 'ms_subclass_90',
       'neighborhood_GrnHill', 'neighborhood_NridgHt', 'neighborhood_StoneBr',
       'condition_2_PosA', 'bldg_type_Duplex', 'roof_style_Mansard',
       'roof_matl_Membran', 'roof_matl_Metal', 'roof_matl_Roll',
       'roof_matl_WdShngl', 'exterior_1st_BrkComm', 'exterior_2nd_Other',
       'heating_GasA', 'heating_GasW', 'heating_Grav', 'heating_OthW',
       'heating_Wall', 'sale_type_Con']
    * corr() function is used to select the continuous independant variable with correlation of + or - 0.5 with saleprice are selected.
      ['overall_qual','kitchen_qual','garage_area','garage_cars','bsmt_qual',
        'year_built','garage_finish','year_remod/add','fireplace_qu','full_bath','foundation_PConc',
         'mas_vnr_area','totrms_abvgrd','fireplaces','heating_qc',                    
        '1st_flr_sf', '2nd_flr_sf', 'bsmt_unf_sf', 'bsmtfin_sf_1',
       'bsmtfin_sf_2', 'exter_qual', 'gr_liv_area']
   * Heatmap is plotted.
     - We could find some of the variables are highly correlated from the heatmap.
      The correlated variables are as follows.
        1.garage_cars and garage_area                                                                                                   
        2.fireplaces aoverall_qualnd fireplace_qu 
        3.Exter_qual and Kitchen_qual
        4.gr_liv_area and totrms_abvgrd
        5.1st_flr_sf and total_bsmt_sf
        6.ms_subclass_90 and bldg_type_duplex
        7.heating_gasA and heating_gasW
      - roof_matl_Metal, roof_matl_Roll and exterior_2nd_Other don't have any values in their training set.  
   
   * Scatter plot is built to check the relation between individual feature with sale price.
     We could see that most of the features are in a linear relationship either positive or negtive with our target saleprice as per the        requirement for multilinear model. But we cannot check the linearity for some categorical variables and some continuous variable like      low qual fin sf, bsmtfin_sf_2 ans bsmt_unf_sf. but those variables are selected by RFECV.
   * Histogram is constructed to check the distribution of the variables involved in the model.From the above histogram we could see our        features have ups and down and they are not normally distributed except garage area, overall_qual, garage_cars and totrms_abvgrd. 
     Sales price is somewhat normally distributed. The sales range is higher in the range 100000 to 300000. So when we train the model will      good to predict the price in this range. Especially, there is sparse data for the houses sold above 500000. So, the model may perform      poorer for house sale price that fall in the range of 500000 to 600000.
   * Function for calculating Adjusted R2. p is the number of predictors used in the model.
   * Function for calculating Root mean square error
   * Created a dataframe to store the results of various models to be constructed and to compare their performance.
   * Initialize the StandardScaler object outside the function so that it can be used while constructing the final model
   * Function created to build a model with selected featureset using linear regression and to check the effect of regularisation.
     This function returns the result of linear regression and regularisation as a column to be added to the Output dataframe
   * Created total of 5 featureset and tested with the model function.
     - featureset1 is same as initial features after RFECV and corr(). This is the 1st set of feature for our model
        featureset1= ['overall_qual','kitchen_qual','garage_area','garage_cars','bsmt_qual',
        'year_built','garage_finish','year_remod/add','fireplace_qu','full_bath','foundation_PConc',
         'mas_vnr_area','totrms_abvgrd','fireplaces','heating_qc',                    
        '1st_flr_sf', '2nd_flr_sf', 'bsmt_unf_sf', 'bsmtfin_sf_1',
       'bsmtfin_sf_2', 'exter_qual', 'gr_liv_area', 'low_qual_fin_sf',
       'total_bsmt_sf', 'ms_subclass_150', 'ms_subclass_90',
       'neighborhood_GrnHill', 'neighborhood_NridgHt', 'neighborhood_StoneBr',
       'condition_2_PosA', 'bldg_type_Duplex', 'roof_style_Mansard',
       'roof_matl_Membran', 'roof_matl_Metal', 'roof_matl_Roll',
       'roof_matl_WdShngl', 'exterior_1st_BrkComm', 'exterior_2nd_Other',
       'heating_GasA', 'heating_GasW', 'heating_Grav', 'heating_OthW',
       'heating_Wall', 'sale_type_Con']
       It is a complex model with 44 features but the accuracy was not bad, The lasso coefficient is used in the upcoming models to
       reduce the features.
       Kaggle RMSE - 32468.64427
       
     - featureset2 I have used the variable which we got as output for RFECV
       featureset2=['1st_flr_sf', '2nd_flr_sf', 'bsmt_unf_sf', 'bsmtfin_sf_1',
       'bsmtfin_sf_2', 'exter_qual', 'gr_liv_area', 'low_qual_fin_sf',
       'total_bsmt_sf', 'ms_subclass_150', 'ms_subclass_90',
       'neighborhood_GrnHill', 'neighborhood_NridgHt', 'neighborhood_StoneBr',
       'condition_2_PosA', 'bldg_type_Duplex', 'roof_style_Mansard',
       'roof_matl_Membran', 'roof_matl_Metal', 'roof_matl_Roll',
       'roof_matl_WdShngl', 'exterior_1st_BrkComm', 'exterior_2nd_Other',
       'heating_GasA', 'heating_GasW', 'heating_Grav', 'heating_OthW',
       'heating_Wall', 'sale_type_Con']
       Model 2 has the least accuracy. They also overfit with the training data. (29 features)
       Kaggle RMSE -37564.62481
       
     - retained one of the highly correlated variables for featureset3
       garage_cars(removed) and garage_area 
       fireplaces(removed) and fireplace_qu 
       overall_qual, Exter_qual(removed) and Kitchen_qual(removed)
       gr_liv_area and totrms_abvgrd(removed)
       1st_flr_sf and total_bsmt_sf (aggregated them instead of removing them)
       ms_subclass_90 and bldg_type_duplex(removed)
       heating_gasA(removed) and heating_gasW
       aggregated the sf for featureset 3
       Now the independant variables in this featureset has less correlation and also have 1 aggregated variable for                              featureset3=['total_sf', 'bsmt_unf_sf', 'bsmtfin_sf_1','bsmtfin_sf_2',  'gr_liv_area', 'low_qual_fin_sf',
        'ms_subclass_150','ms_subclass_90','overall_qual','neighborhood_GrnHill', 'neighborhood_NridgHt','neighborhood_StoneBr',
       'condition_2_PosA',  'roof_style_Mansard','roof_matl_Membran', 'roof_matl_Metal', 'roof_matl_Roll',
       'roof_matl_WdShngl', 'exterior_1st_BrkComm', 'exterior_2nd_Other','heating_GasW', 'heating_Grav', 'heating_OthW',
       'heating_Wall', 'sale_type_Con']
       Model 3 is better than model2 but also model2 also overfit with the training data.
       Kaggle RMSE -33087.02166
       
      -Based on the lasso coefficient we will remove features with 0 coefficient for featureset4. Also removing 2nd flr_sf as its                  correlation with sales price is less compared to other variables. This time aggregated only 1st_flr_sf and total_bsmt_sf.   
       creating the aggregate for 1st flr sf and total bsmt sf
       variables with 0 or very low lasso coefficient for featureset1 is removed in the set
       featureset4= ['1st_flr_sf+total_bsmt_sf','overall_qual','garage_area','year_built','year_remod/add',
       'fireplace_qu','full_bath','mas_vnr_area','heating_qc','bsmtfin_sf_1', 'gr_liv_area',
       'low_qual_fin_sf', 'ms_subclass_90','neighborhood_GrnHill', 'neighborhood_NridgHt',
       'neighborhood_StoneBr','condition_2_PosA','roof_style_Mansard','roof_matl_Membran',
       'roof_matl_WdShngl','exterior_1st_BrkComm','heating_GasA', 'heating_OthW',
       'heating_Wall','sale_type_Con']
       Compared to model1, model 4 has slightly lesser accuracy but still the code is simpler(only 25 features) than model1 and has better        accuracy compared to Model 2 & 3.   
       Kaggle RMSE -32553.17515
       
       -heating_GasA', 'heating_OthW','exterior_1st_BrkComm''condition_2_PosA','roof_style_Mansard',
        'roof_matl_WdShngl','roof_matl_Membran', affects the saleprice very less compared 
         to the other variables based on the lasso coefficient for featureset1 are removed.
        (Also tested after removing each variable one at a time).
        kitchen_qual and 'exter_qual' which are well correlated with saleprice are added.
        featureset5= ['1st_flr_sf+total_bsmt_sf','overall_qual','garage_area','year_built','fireplace_qu',
        'full_bath','mas_vnr_area','heating_qc','bsmtfin_sf_1', 'gr_liv_area', 'kitchen_qual',
        'ms_subclass_90','year_remod/add','neighborhood_GrnHill', 'neighborhood_NridgHt',
        'neighborhood_StoneBr', 'heating_Wall','sale_type_Con','low_qual_fin_sf','exter_qual']
        Model 5 and Final model has better accuracy than any other model.   The ridge adjusted R2 is 0.870895 for test data is higher than         the ridge score for train data 0.885387. So, The model also does not overfit with the training data. So, The model also does not           overfit with the training data. This model has only 20 features. It is simple compared to the other model. 
        Kaggle RMSE-31386.30343
 
 ## MODELING AND MODEL EVALUATION
 
   * Our X valve is our independant variables stored under features. Y value or our target is saleprice.
   * As per industrial norm the train and test data is split into 80:20%
   * From the above feature selection, We chose Featureset5 for our model and ridge regression as the regressor.
   * Initialize the StandardScaler object
   * Finding the optimal ridge alpha
   * Finding the ridge score for the train data
   * Print the ridge intercept
   * Dataframe created to store the ridge coefficient
   * Plot to visualize the features involved in the model and effect of its coefficient
      
      ***Variables with positive relation with saleprice***:Price increase with increase in the variables
         gr_liv_area, overall qual, bsmtfin_sf_1, 1st_flr_sf+total_bsmt_sf,exter_qual,kitchen_qual, neighbourhood_NridgHt, garage_area,              neighbourhood_StoneBr,year_built,year_remod/add, mas_vnr_area, fireplace_qu,heating_qc,neighbourhood_grnHill,heating_Wall and             sale_typ_Con

      ***Variables with negative relation with saleprice***:Price decrease with increase in the variables
         full_bath, low_qual_fin_sf, and ms_subclass_90.
   * Prediction of sales is executed
   * Metrics Adjusted R sqr and RMSE is calculated which is 0.8853868342510992 and is 25499.711592683936 respectively.
   * sns regplot is plotted between the Test price and Predicted price
   * The Predicted sales price have a linear relationship with y-test except for few outliers.
   * Histogram plotted to check the Residual distribution
   * The residual plot shows a normal distribution. This shows about 100 observation show a 0 residual and the residual goes down from          there. But there is a outlier with residual of about 170000. This is due to outliers in our data.
   * Scatter plot plotted between Prediction and Residual
   * The above plot shows less variance for smaller value of sales.While the variance is higher and most of the points are above line which      shows the predicted price is lower than the actual sales for most of the sales more than USD 300000. But the model is Homoskedastic as      per the requirement for linear model.
   
## PREDICTION WITH TEST DATA 

   * Aggregating 1st_flr_sf and total_bsmt_sf as perour model 5 with featureset5
   * Histogram plotted for test data. The above graph illustrate the spread of test data. The data has an uneven spread.
   * Standardisation and predicting with ridge model.
   * A column is created in the df_tst for storing the predicted list
   * The result is rounded to 2 decimals
   * Dataframe df_result is created to store the predicted sales with 2 decimal points and its id
   * id is converted into index
   * The dataframe is converted into csv file for submitting the result to csv
     
   
## RECOMMENDATION AND CONCLUSION

   We have created a multi linear model for our sales price prediction. We have used Ridge regression with RMSE of 25499.7 which is lower     than the linear regression model 25516.4. The Kaggle RMSE for the model RMSE-31386.30.The features that highly affect the sales price       includes gr_liv_area, overall qual, bsmtfin_sf_1, 1st_flr_sf+total_bsmt_sf,exter_qual,kitchen_qual, neighbourhood_NridgHt, garage_area,     neighbourhood_StoneBr,year_built,year_remod/add,mas_vnr_area,fireplace_qu,heating_qc,neighbourhood_grnHill,heating_Wall and                 sale_typ_Con shows a positive linear relationship with saleprice. While there is a decrease in sales price with increase in full_bath,     low_qual_fin_sf, and ms_subclass_90. 

  This model works well for sale price until 500000 with Root mean square error of about USD 25499.7 for house at AMES. The predicted         saleprice have linear relationship, The Residual histogram follow a normal distribution and the Residual Vs Test Sale price the values     are spread around zero line and it is homoskedastic as per our requirement for linear model, But the plot have one outlier which shows     it is only trained until sales of USD 500000. 

  The model is quite helpful as most of AMES houses are sold at range of USD 70000 to USD 350000.
