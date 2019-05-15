library(dplyr)
library(ggplot2)
library(caret)
library(tidyr)
library(gridExtra)
# load data 
setwd("./documents")
data <- read.csv("test_campaign.csv")
summary(data)
# This is a large dataset, so instead of imputing NA values, we'll simply delete rows with NA values (n = 74)
data_clean <- na.omit(data)
# another problem shown in this is click have a quite abnormal value, it's apparantly an error, we should delete it 
data_clean <- filter(data_clean, !grepl(5748394, click))
data_clean$click <- as.factor(data_clean$click)
# First, we need to select the relevant features for prediction. It's a big dataset, good to build a prediction model.
# considering computational cost and overfitting rist, features need to be reduced.

# first, let's use some domain knowledge to eliminate features 
# To begin with, I assume that geographic information is unnecesary
# first let's look at the geographic feautre. 

#build a dataframe with only geographic features, and click action 
geo <- select(data_clean,geo_dma, geo_region, geo_postal_code, geo_city, click)
geo <- group_by(geo, geo$click)

# Since i'm working on this case wih a macbook air, with limited computational power. I'll not plot all the features 
# Instead I'll do simple statistical study with the summarise function 
summarise(geo, mean(geo_dma), mean(geo_region), mean(geo_postal_code), mean(geo_city))
summarise(geo, sd(geo_dma), sd(geo_region), sd(geo_postal_code), sd(geo_city))

#Judging from the result, I assume that the geographic information is not relevant for prediction, 
#since they are practically the same between two categoties 0/1
# now let's visualize those features to verify if there is a pattern in their distriution 

p_dma <- ggplot(data = geo) + geom_boxplot(mapping = aes(y = geo$geo_dma, x = geo$click)) + labs(x = "click", y = "DMA")
p_region <- ggplot(data = geo) + geom_boxplot(mapping = aes(y = geo$geo_region, x = geo$click))+ labs(x = "click", y = "region")
p_postalcode <- ggplot(data = geo) + geom_boxplot(mapping = aes(y = geo$geo_postal_code, x = geo$click))+ labs(x = "click", y = "postal code")
p_city <- ggplot(data = geo) + geom_boxplot(mapping = aes(y = geo$geo_city, x = geo$click))+ labs(x = "click", y = "city")
grid.arrange(p_dma, p_region, p_postalcode, p_city)

# we can see that there are some outliers in dma, region and city. 
# but the boxplot shows that the action click doesn't really have an inlfuence on the distribution of geographic information
# therefore they should be dropped 
data_clean <- select(data_clean, -starts_with("geo"))

#Now,let's eliminate features that shows trival statistical significance 
# when we look at other features, we also need to look out for near zero variables, they will interfere with our model prediction 
nearZeroVar(data_clean, saveMetrics = TRUE)
# so we need to consider to drop creative_id to improve our prediction
# however, maybe the advertisement identity is important information, this need to be discussed with clients
data_clean <- select(data_clean, -creative_id)

# now let's look for correlations, if yes, maybe it should be dropped 
cor <- cor(data_clean[,1:10])
hcor <- findCorrelation(cor, cutoff=0.8, verbose = TRUE)

# judging from the code-book, these two features are clearly similar
# there column 9 will be dropped 
data_clean <- data_clean[, -hcor]

# names of the class levels should be converted to valid names 
levels(data_clean$click) <- make.names(levels(factor(data_clean$click)))
# features can still be dropped during the training process, with varImp function 
#now let's build our prediction model 
# this is clearly a classification problem, try to predict two classes click = "0" or "1"
prop.table(table(data_clean$click))
#the result shows that this is a highly imbalanced binary classification problem 
# we need to keep this in mind when choosing algorithms 
# for example the error measure will not be (R)MSE, but should be accuracy and Kappa. 
# tree algorithms are usually better suited for imbalanced data classification 

# now let's look at the data distribution 
sapply(data_clean,sd)
sapply(data_clean,class)
# some columns are quite skewed, we need to do some standarization 
# first we need to split the data 
# we are going to split the data into two parts, train dataset (70%) and test dataset(30%)
# in the train dataset, we'll use k folds cross validation
inTrain <- createDataPartition(y = data_clean$click, p = 0.7, list = FALSE )
training <- data_clean[inTrain,]; testing <- data_clean[-inTrain,]

# In order to deal with imbalanced data, several approches can be used 
# we can change sampling, use cost sensitivy learning, change error measurement 
# here we are just going to do the simplest, fastest and maybe most efficient, sampling control 
# we are going to use 10 folds CV, and down sampling to deal with imbalanced data 
# this may cause prediction bias
# another solution may be change the performance matrix 
# Here we use a performance matrix AUC, which calclute the deritive of the ROC curve 
## For accuracy, Kappa, the area under the ROC curve,
## sensitivity and specificity:
fiveStats <- function(...) 
    c(twoClassSummary(...), 
      defaultSummary(...))
## Everything but the area under the ROC curve:
fourStats <- function (data, lev = levels(data$obs), model = NULL)
{
       accKapp <- postResample(data[, "pred"], data[, "obs"])
       out <- c(accKapp,
         sensitivity(data[, "pred"], data[, "obs"], lev[1]),
         specificity(data[, "pred"], data[, "obs"], lev[2]))
         names(out)[3:4] <- c("Sens", "Spec")
        out
}
# two different smapling methods 
control_down <- trainControl(method = "cv",
                        number = 10,
                        sampling = "down",
                        verboseIter = TRUE,
                        summaryFunction = fiveStats,
                        classProbs = TRUE)
control_smote <- trainControl(method = "cv",
                             number = 10,
                             sampling = "smote",
                             verboseIter = TRUE,
                             summaryFunction = fiveStats,
                             classProbs = TRUE)
control_noprob <- trainControl(method = "cv",
                              number = 10,
                              sampling = "down",
                              verboseIter = TRUE,
                              summaryFunction = fourStats,
                              classProbs = FALSE)

# due to my computational limit, i'll choose not to do bagging or bootstraping. 
# but this is a nice way to smooth the result  
# for this kind of binary classification problem, expecially with a imbalanced dataset
# tree classification is usually good, I'll skip random forest, because it will take hours of training 

#lda 
fit.lda <- train(click ~., data = training, method = "lda", 
                   preProcess = c("center", "scale"), 
                   metric = "ROC", 
                tuneLength = 5,
                   trControl = control_down )
pred_lda <- predict(fit.lda, testing)
con_lda <- confusionMatrix(pred_lda, testing$click)
# rpart
fit.rpart <- train(click ~., data = training, method = "rpart", 
                   preProcess = c("center", "scale"), 
                   metric = "ROC", 
                   tuneLength = 5,
                   trControl = control_down)
pred_rpart <- predict(fit.rpart, testing)
con_rpart <- confusionMatrix(pred_rpart, testing$click)
#gbm
fit.gbm <- train(click ~., data = training, method = "gbm", 
                   preProcess = c("center", "scale"), 
                   metric = "ROC", 
                   tuneLength = 5,
                   trControl = control_down)
pred_gbm <- predict(fit.rpart, testing)
con_gbm <- confusionMatrix(pred_gbm, testing$click)

pred_results <- data.frame(click = testing$click)
pred_results$lda <- predict(fit.lda, testing, type = "prob")[,1]
pred_results$gbm <- predict(fit.gbm, testing, type = "prob")[,1]
pred_results$rpart <- predict(fit.rpart, testing, type = "prob")[,1]

ldaROC <- roc(pred_results$click, pred_results$lda, 
              levels = rev(levels(pred_results$click)))

ldaThresh <- coords(ldaROC, x = "best", best.method = "closest.topleft")


newValue <- factor(ifelse(pred_results$lda > rfThresh, "X0", "X1"), levels = levels(pred_results$click))
#rpartcost
fit.rpartcost <- train(click ~., data = training, 
                method = "rpart", 
                preProcess = c("center", "scale"), 
                metric = "Kappa", 
                parms = list(loss = matrix(c(0,10,1,0), nrow = 2)),
                trControl = control_noprob)
                
pred_rpartcost <- predict(fit.rpartcost, testing)
con_rpartcost <- confusionMatrix(pred_rpartcost, testing$click)



