# load libraries
library(doParallel)  # for quicker training using parallel processing
library(caret)


# read csvs
train_raw <- read.csv('C:/Rworkspace/pml/project/pml-training.csv', na.strings=c('NA', '#DIV/0!'))
test_raw <- read.csv('C:/Rworkspace/pml/project/pml-testing.csv', na.strings=c('NA', '#DIV/0!'))


# reduce rows and columns
na_columns_index <- which(sapply(test_raw, function(x) all(is.na(x))))
train_no_na_columns <- train_raw[,-na_columns_index]
test_no_na_columns <- test_raw[,-na_columns_index]

new_window_is_no_rows <- which(train_no_na_columns$new_window == 'no')
train_no_na_columns <- train_no_na_columns[new_window_is_no_rows,]

first_columns <- 1:6
train_no_na_no_first <- train_no_na_columns[,-first_columns]
test_no_na_no_first <- test_no_na_columns[,-first_columns]

train_all <- train_no_na_no_first
test <- test_no_na_no_first


# split data
train_index <- createDataPartition(train_all$classe, p=0.7, list=FALSE)
train <- train_all[train_index,]
train_test <- train_all[-train_index,]


# make and register cluster
cluster <- makeCluster(detectCores())
registerDoParallel(cluster)


# fit random forest
control <- trainControl(method="cv", number=4)
Sys.time()
fit_rf <- train(classe~., data=train, method='rf', trControl=control)
# > fit_rf
# Random Forest 
# 
# 13453 samples
# 53 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E' 
# 
# No pre-processing
# Resampling: Cross-Validated (4 fold) 
# 
# Summary of sample sizes: 10090, 10090, 10089, 10090 
# 
# Resampling results across tuning parameters:
#   
#   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
# 2    0.993     0.991  0.00342      0.00433 
# 27    0.997     0.996  0.00176      0.00222 
# 53    0.995     0.994  0.00268      0.00340 
# 
# Accuracy was used to select the optimal model using  the largest value.
# The final value used for the model was mtry = 27.


# validation set prediction accuracy
predictions_train_test <- predict(fit_rf, newdata=train_test)
confusionMatrix(predictions_train_test, train_test$classe)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    A    B    C    D    E
# A 1641   10    0    0    0
# B    0 1099    4    0    0
# C    0    2 1000    7    0
# D    0    4    1  937    1
# E    0    0    0    0 1057
# 
# Overall Statistics
# 
# Accuracy : 0.995           
# 95% CI : (0.9928, 0.9966)
# No Information Rate : 0.2847          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.9936          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            1.0000   0.9857   0.9950   0.9926   0.9991
# Specificity            0.9976   0.9991   0.9981   0.9988   1.0000
# Pos Pred Value         0.9939   0.9964   0.9911   0.9936   1.0000
# Neg Pred Value         1.0000   0.9966   0.9989   0.9985   0.9998
# Prevalence             0.2847   0.1935   0.1744   0.1638   0.1836
# Detection Rate         0.2847   0.1907   0.1735   0.1626   0.1834
# Detection Prevalence   0.2865   0.1914   0.1751   0.1636   0.1834
# Balanced Accuracy      0.9988   0.9924   0.9966   0.9957   0.9995


# predict test data
predictions_test <- predict(fit_rf, newdata=test)
# > predictions_test
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E


# create submission files
answers <- as.character(predictions_test)
n = length(answers)
for(i in 1:n) {
  filename = paste0("problem_id_",i,".txt")
  write.table(answers[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}





