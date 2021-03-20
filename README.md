# House Prices

## Data description
This datasets consists of 1460 rows and 81 features, which 43 is categorical and the other 38 is numerical.
The target is to predict the SalePrice feature which is the property's sale price in dollars.

box plot: YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF

TODO: drop ID, Alley,
MSSubClass not converted 
r2_score ??

## Data Pre-processing

The first step is to find the percentange of missing values in every feature in order to delete some feautres that they have many null values.

Id - 0%
MSSubClass - 0%  
MSZoning - 0%    
<span style="color:red"> LotFrontage - 18% </span>
LotArea - 0%     
Street - 0%      
<span style="color:red"> Alley - 94% </span>
LotShape - 0%    
LandContour - 0% 
Utilities - 0%
LotConfig - 0%
LandSlope - 0%
Neighborhood - 0%
Condition1 - 0%
Condition2 - 0%
BldgType - 0%
HouseStyle - 0%
OverallQual - 0%
OverallCond - 0%
YearBuilt - 0%
YearRemodAdd - 0%
RoofStyle - 0%
RoofMatl - 0%
Exterior1st - 0%
Exterior2nd - 0%
MasVnrType - 1%
MasVnrArea - 1%
ExterQual - 0%
ExterCond - 0%
Foundation - 0%
BsmtQual - 3%
BsmtCond - 3%
BsmtExposure - 3%
BsmtFinType1 - 3%
BsmtFinSF1 - 0%
BsmtFinType2 - 3%
BsmtFinSF2 - 0%
BsmtUnfSF - 0%
TotalBsmtSF - 0%
Heating - 0%
HeatingQC - 0%
CentralAir - 0%
Electrical - 0%
1stFlrSF - 0%
2ndFlrSF - 0%
LowQualFinSF - 0%
GrLivArea - 0%
BsmtFullBath - 0%
BsmtHalfBath - 0%
FullBath - 0%
HalfBath - 0%
BedroomAbvGr - 0%
KitchenAbvGr - 0%
KitchenQual - 0%
TotRmsAbvGrd - 0%
Functional - 0%
Fireplaces - 0%
<span style="color:red"> FireplaceQu - 47% </span>
GarageType - 6%
GarageYrBlt - 6%
GarageFinish - 6%
GarageCars - 0%
GarageArea - 0%
GarageQual - 6%
GarageCond - 6%
PavedDrive - 0%
WoodDeckSF - 0%
OpenPorchSF - 0%
EnclosedPorch - 0%
3SsnPorch - 0%
ScreenPorch - 0%
PoolArea - 0%
<span style="color:red"> PoolQC - 100% </span>
<span style="color:red"> Fence - 81% </span>
<span style="color:red"> MiscFeature - 96% </span>
MiscVal - 0%
MoSold - 0%
YrSold - 0%
SaleType - 0%
SaleCondition - 0%
SalePrice - 0%


# Regression