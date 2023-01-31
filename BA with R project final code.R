
# Detecting Credit Card Fraud

# 1. Prepare Problem

# a) Load libraries
install.packages("smotefamily")
install.packages("e1071")
install.packages("caTools")
install.packages("class")
install.packages("rlang")
install.packages("caret")
install.packages("e1071")
install.packages("class")
install.packages("neuralnet")

library(neuralnet)
library(rlang)
library(e1071)
library(caTools)
library(class)
library(dplyr) # for data manipulation
library(stringr) # for data manipulation
library(caret) # for sampling
library(caTools) # for train/test split
library(ggplot2) # for data visualization
library(corrplot) # for correlations
library(Rtsne) # for tsne plotting
library(ROSE)# for ROSE sampling
library(rpart)# for decision tree model
library(Rborist)# for random forest model
library(xgboost) # for xgboost model

# b) Load dataset
df<- read.csv("creditcard.csv")
df<- df[sample(nrow(df), 10000), ]

# 2. Summarize Data

# a) Descriptive statistics

# Data Exploration
dim(df)
head(df,6)
tail(df,6)
table(df$Class)
summary(df$Amount)
names(df)
var(df$Amount)
sd(df$Amount)


# b) Data visualizations

# Distribution of class labels
common_theme <- theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data = df, aes(x = factor(Class), 
                      y = prop.table(stat(count)), fill = factor(Class),
                      label = scales::percent(prop.table(stat(count))))) +
  geom_bar(position = "dodge") + 
  geom_text(stat = 'count',
            position = position_dodge(.9), 
            vjust = -0.5, 
            size = 3) + 
  scale_x_discrete(labels = c("no fraud", "fraud"))+
  scale_y_continuous(labels = scales::percent)+
  labs(x = 'Class', y = 'Percentage') +
  ggtitle("Distribution of class labels") +
  common_theme


# Distribution of variable 'Time' by class
df %>%
  ggplot(aes(x = Time, fill = factor(Class))) + geom_histogram(bins = 100)+
  labs(x = 'Time in seconds since first transaction', y = 'No. of transactions') +
  ggtitle('Distribution of time of transaction by class') +
  facet_grid(Class ~ ., scales = 'free_y') + common_theme


# Distribution of transaction amount by class
ggplot(df, aes(x = factor(Class), y = Amount)) + geom_boxplot() + 
  labs(x = 'Class', y = 'Amount') +
  ggtitle("Distribution of transaction amount by class") + common_theme

# Feature Selection
correlations <- cor(df[,-1],method="pearson")
corrplot(correlations, number.cex = .9, method = "circle", type = "full", tl.cex=0.8,tl.col = "black")

# 3. Prepare Data

# a) Data Cleaning

# checking missing values
colSums(is.na(df))

# Remove 'Time' variable
df <- df[,-1]

# Change 'Class' variable to factor
df$Class <- as.factor(df$Class)
levels(df$Class) <- c("Not_Fraud", "Fraud")

head(df)
# Scale numeric variables
df[,-30] <- scale(df[,-30])
head(df)


# 4. Evaluate Algorithms
# a) Split-out validation dataset

set.seed(123)
split <- sample.split(df$Class, SplitRatio = 0.7)
train <-  subset(df, split == TRUE)
test <- subset(df, split == FALSE)

# b) Test options and evaluation metric

# class ratio initially
table(train$Class)

# downsampling
set.seed(9560)
down_train <- downSample(x = train[, -ncol(train)],
                         y = train$Class)
table(down_train$Class)

# Build down-sampled model
set.seed(5627)
down_fit <- rpart(Class ~ ., data = down_train)

# AUC on down-sampled data
pred_down <- predict(down_fit, newdata = test)
roc.curve(test$Class, pred_down[,2], plotit = FALSE)

# upsampling
set.seed(9560)
up_train <- upSample(x = train[, -ncol(train)],
                     y = train$Class)
table(up_train$Class)

# Build up-sampled model
set.seed(5627)
up_fit <- rpart(Class ~ ., data = up_train)

# AUC on up-sampled data
pred_up <- predict(up_fit, newdata = test)
roc.curve(test$Class, pred_up[,2], plotit = FALSE)

# rose
set.seed(9560)
rose_train <- ROSE(Class ~ ., data  = train)$data
table(rose_train$Class)

# Build rose model
set.seed(5627)
rose_fit <- rpart(Class ~ ., data = rose_train)

# AUC on ROSE fit data
pred_rose <- predict(rose_fit, newdata = test)
roc.curve(test$Class, pred_rose[,2], plotit = FALSE)

#6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use

# Decision Tree
set.seed(5627)
orig_fit <- rpart(Class ~ ., data = train)
pred_orig <- predict(orig_fit, newdata = test, method = "class")
roc.curve(test$Class, pred_orig[,2], plotit = TRUE)

# Logistic Regression
x = up_train[, -30]
y = up_train[,30]
model <- glm(Class ~.,family=binomial(link='logit'),data=up_train)
summary(model)
pred_glm = predict(model, newdata = test)
cm = table(test$Class, pred_glm)
roc.curve(test$Class, pred_glm, plotit = TRUE)

# Random Forest
rf_fit <- Rborist(x, y, ntree = 1000, minNode = 20, maxLeaf = 13)
rf_pred <- predict(rf_fit, test[,-30], ctgCensus = "prob")
prob <- rf_pred$prob
roc.curve(test$Class, prob[,2], plotit = TRUE)

# SVM
labels <- up_train$Class # Convert class labels from factor to numeric
y <- recode(labels, 'Not_Fraud' = 0, "Fraud" = 1)
classifier = svm(formula = Class ~ .,
                 data = up_train,
                 type = 'C-classification',
                 kernel = 'linear')
classifier
y_pred = predict(classifier, newdata = test)
cm = table(test$Class, y_pred)
cm
roc.curve(test$Class, y_pred, plotit = TRUE)

# XGB
set.seed(42)
xgb <- xgboost(data = data.matrix(up_train[,-30]), 
               label = y,
               eta = 0.1,
               gamma = 0.1,
               max_depth = 10, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.6,
               verbose = 0,
               nthread = 7,
)
xgb_pred <- predict(xgb, data.matrix(test[,-30]))
roc.curve(test$Class, xgb_pred, plotit = TRUE)

names <- dimnames(data.matrix(up_train[,-30]))[[2]]
importance_matrix <- xgb.importance(names, model = xgb) # Compute feature importance matrix
xgb.plot.importance(importance_matrix[1:10,]) # Nice graph

# Neural Network
ANN_model =neuralnet(Class~.,up_train,linear.output=FALSE)
plot(ANN_model)
predANN=compute(ANN_model,test)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)
