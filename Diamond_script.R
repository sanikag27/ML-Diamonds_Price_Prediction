## Installing and loading the necesarry libraries


suppressMessages(library(tidyverse)) 
suppressMessages(library(caret))    
suppressMessages(library(randomForest))
suppressMessages(library(pROC))


## Loading and Preprocessing the data

diamonds <- read.csv("diamonds.csv")

cat("Structure of the diamonds dataset:\n")
str(diamonds)
cat("\nFirst few rows of the dataset:\n")
head(diamonds)



#converting the categorical data into factors

diamonds$cut <- factor(diamonds$cut, levels = c("Fair", "Good", "Very Good", "Premium", "Ideal"))
diamonds$color <- factor(diamonds$color, levels = c("J", "I", "H", "G", "F", "E", "D"))
diamonds$clarity <- factor(diamonds$clarity, levels = c("I1", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2", "IF"))


# Apply log in Price

diamonds$price <- log(diamonds$price)

cat("Number of missing values per column:\n")
print(colSums(is.na(diamonds)))

# shows that there are no missing values


## Splitting data into test and training sets

set.seed(123)  
trainIndex <- createDataPartition(diamonds$price, p = 0.8, list = FALSE)
trainData <- diamonds[trainIndex, ]
testData <- diamonds[-trainIndex, ]

cat("Training set size: ", nrow(trainData), "\n")
cat("Test set size: ", nrow(testData), "\n")



# Visualizing Predictor Distributions
library(ggplot2)
library(gridExtra)

# Facet plot: Predictor distributions (histograms)
histogram_plot <- trainData %>%
  pivot_longer(cols = c(carat, depth, table, x, y, z), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Value)) +
  geom_histogram(binwidth = 0.5, fill = "brown", color = "white") +
  facet_wrap(~Variable, scales = "free", ncol = 3) +
  labs(title = "Predictor Distributions",
       x = "Value",
       y = "Frequency") +
  theme_minimal()

# Print the histogram plot
print(histogram_plot)

# Facet plot: Predictor-response relationships (box plots)
boxplot_plot <- trainData %>%
  pivot_longer(cols = c(carat, depth, table, x, y, z), names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = Variable, y = price)) +
  geom_boxplot(fill = "indianred") +
  facet_wrap(~Variable, scales = "free", ncol = 3) +
  labs(title = "Predictor-Response Relationships",
       x = "Predictor",
       y = "Log-Transformed Price") +
  theme_minimal()

# Print the box plot


### Starting first with linear methods

## Linear Regression 

# Training a linear regression model
model_lr <- lm(price ~ carat + cut + color + clarity + x + y + z + table + depth, data = trainData)
cat("Linear Regression Model Summary:\n")
summary(model_lr)
average_price <- mean(diamonds$price)
cat("The average price of diamonds is:", average_price, "\n")

# predictions on the test set
predictions <- predict(model_lr, newdata = testData)
cat("Model Performance Metrics:\n")


rmse <- signif(sqrt(mean((predictions - testData$price)^2)), 3)
cat("Root Mean Squared Error (RMSE): ", rmse, "\n")
rsq <- signif(cor(predictions, testData$price)^2, 3)
cat("R-squared: ", rsq, "\n")

# Plot actual vs predicted prices
plot(predictions, testData$price, main = "Predicted vs Actual Prices", 
     xlab = "Predicted Price", ylab = "Actual Price", col = "brown2", pch = 16)
abline(0, 1, col = "brown4")

## Random Forest 

# Training a Random Forest model 
model_rf <- randomForest(price ~ carat + cut + color + clarity + x + y + z + table + depth, ntree= 50,
                         data = trainData)

# print(model_rf)
cat("Random Forest Model Summary:\n")
print(model_rf)
cat("Feature Importance:\n")
print(model_rf$importance)


# Note: this was taking forever, so using ntree = 50

# Making predictions on the test set
predictions <- predict(model_rf, newdata = testData)

cat("First few predictions vs actual values:\n")
comparison <- data.frame(Actual = testData$price, Predicted = predictions)
print(head(comparison))

# Calculating the performance metrics: RMSE, R-squared
rmse_rf <- signif(sqrt(mean((predictions - testData$price)^2)), 3)
rsq <- signif(cor(predictions, testData$price)^2, 3)
cat("Root Mean Squared Error (RMSE):", rmse_rf, "\n")
cat("R-squared:", rsq, "\n")

# Visualizing predictions vs actual values
plot(predictions, testData$price, main = "Predicted vs Actual Prices", 
     xlab = "Predicted Price", ylab = "Actual Price", col = "brown2", pch = 16)
abline(0, 1, col = "brown4")

# Residual plot
residuals <- predictions - testData$price
plot(predictions, residuals, main = "Residuals Plot",
     xlab = "Predicted Price", ylab = "Residuals", col = "brown2", pch = 16)
abline(h = 0, col = "maroon")

# QQ plot 
qqnorm(residuals, main = "QQ Plot of Residuals")
qqline(residuals, col = "indianred")

#multicollinrarity 
library(car)
vif(model_lr) #vif >5 would indicate mullticollinearity


## Decision Tree Model

library(rpart)
library(rpart.plot)
# Train the decision tree model
model_tree <- rpart(price ~ carat + cut + color + clarity + x + y + z + table + depth, 
                    data = trainData, 
                    method = "anova")

# Summary of the decision tree model
cat("Decision Tree Model Summary:\n")
print(model_tree)
rpart.plot(model_tree, type = 2, fallen.leaves = TRUE, main = "Decision Tree for Diamond Prices")

# Predictions
predictions_tree <- predict(model_tree, newdata = testData)

# Performance Metrics
rmse_tree <- signif(sqrt(mean((predictions_tree - testData$price)^2)), 3)
rsq_tree <- signif(cor(predictions_tree, testData$price)^2, 3)

cat("Decision Tree RMSE:", rmse_tree, "\n")
cat("Decision Tree R-squared:", rsq_tree, "\n")

# Comparing it to average price, this is an okay model but we could do better.

## Current xgboost

# install.packages("xgboost")
library(xgboost)
# Convert categorical variables to numeric
trainData_xgb <- trainData %>%
  mutate(across(where(is.factor), as.numeric))
testData_xgb <- testData %>%
  mutate(across(where(is.factor), as.numeric))

# Prepare matrix format
train_matrix <- as.matrix(trainData_xgb[, -which(names(trainData_xgb) == "price")])
train_labels <- trainData_xgb$price
test_matrix <- as.matrix(testData_xgb[, -which(names(testData_xgb) == "price")])
test_labels <- testData_xgb$price

# Train the model
model_xgb <- xgboost(data = train_matrix, label = train_labels, 
                     objective = "reg:squarederror", 
                     nrounds = 100, verbose = 0)


# Predictions
predictions_xgb <- predict(model_xgb, newdata = test_matrix)

# Performance Metrics
rmse_xgb <- signif(sqrt(mean((predictions_xgb - test_labels)^2)), 3)
rsq_xgb <- signif(cor(predictions_xgb, test_labels)^2, 3)

cat("XGBoost RMSE:", rmse_xgb, "\n")
cat("XGBoost R-squared:", rsq_xgb, "\n")

## Class XGBoost

suppressMessages(library(xgboost))
suppressMessages(library(tidyverse)) 
# Loading Data
df <- read.csv("diamonds.csv")

#converting the categorical data into factors
df$cut <- as.factor(df$cut)
df$color <- as.factor(df$color)
df$clarity <- as.factor(df$clarity)

# Apply log in Price
df$price <- log(df$price)

# Remove Index
df %>% select(., -X) -> df

# Splitting Data again
set.seed(123)
s <- sample(nrow(df),round(0.7*nrow(df)))
df.train <- df[s,]
df.test  <- df[-s,]

# Convert categorical variables to numeric
trainData_xgb <- df.train %>%
  mutate(across(where(is.factor), as.numeric))
testData_xgb <- df.test %>%
  mutate(across(where(is.factor), as.numeric))

resp.train <- trainData_xgb %>% select(., price) %>% pull(.)
resp.test  <- testData_xgb %>% select(., price) %>% pull(.)
pred.train <- trainData_xgb %>% select(., -price)
pred.test  <- testData_xgb %>% select(., -price)

train <- xgb.DMatrix(data=as.matrix(pred.train),label=resp.train)
test <- xgb.DMatrix(data=as.matrix(pred.test),label=resp.test)
xgb.cv.out <- xgb.cv(params=list(objective="reg:squarederror"),train,nrounds=2000,nfold=5,verbose=0)
rmse.min <- xgb.cv.out$evaluation_log$test_rmse_mean
cat("The optimal number of trees is ",which.min(rmse.min),"\n")

xgb.out <- xgboost(train,nrounds=which.min(rmse.min),params=list(objective="reg:squarederror"),verbose=0)
resp.pred <- predict(xgb.out,newdata=test)

# Performance Metrics
rmse_xgb <- signif(sqrt(mean((resp.pred - resp.test)^2)), 3)
rsq_xgb <- signif(cor(resp.pred, resp.test)^2, 3)

cat("XGBoost RMSE:", rmse_xgb, "\n")
cat("XGBoost R-squared:", rsq_xgb, "\n")



plot(resp.test, resp.pred, main = "Predicted vs Actual Values",
     xlab = "Actual Price", ylab = "Predicted Price", col = "brown2", pch = 16)
abline(0, 1, col = "brown4")  # 45-degree line for comparison

# Adjusting the appearance of the importance plot
library(xgboost)
library(ggplot2)

# Get feature importance
imp.out <- xgb.importance(model = xgb.out)

# Convert to a data frame for ggplot
imp.data <- as.data.frame(imp.out)

# Create a ggplot bar plot for feature importance
ggplot(imp.data, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#a50f15", color = "black", width = 0.7) +
  coord_flip() +  # Flip coordinates to make horizontal bars
  labs(
    title = "Feature Importance",
    x = "Feature",
    y = "Importance (Gain)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(size = 10)
  )

ggsave("feature_importance_plot.png", dpi = 300, width = 6, height = 4)

library(ggplot2)

# Combine actual and predicted values into a data frame
plot_data <- data.frame(
  ActualPrice = resp.test,
  PredictedPrice = resp.pred
)

# Create the plot
ggplot(plot_data, aes(x = ActualPrice, y = PredictedPrice)) +
  geom_point(color = "#a50f15", alpha = 0.6, size = 2) +  # Scatter plot points
  geom_abline(intercept = 0, slope = 1, color = "#67000d", linetype = "dashed", size = 1) +  # 45-degree line
  labs(
    title = "Predicted vs Actual Values",
    x = "Actual Price",
    y = "Predicted Price"
  ) +
  theme_minimal(base_size = 14) +  # Clean, minimal theme
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),  # Center-align title
    axis.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray85"),  # Subtle gridlines
    panel.grid.minor = element_blank()  # Remove minor gridlines
  )

ggsave("predicted_vs_actual.png", width = 8, height = 6, dpi = 300)

#log transformations
model_lr_transformed <- lm(log(price) ~ carat + cut + color + clarity + x + y + z + table + depth, data = trainData)
# Original distribution of price
hist(trainData$price, main = "Original Price Distribution", xlab = "Price", col = "coral3", border = "brown4")

# Log-transformed distribution of price
hist(log(trainData$price), main = "Log-Transformed Price Distribution", xlab = "Log(Price)", col = "coral3", border = "brown4")

summary(model_lr) #predictors with p values > 0.05 can be removed. 


### Best subset model selection

# Preprocessing
testData_bss<- testData %>% rename("Y"="y")
testData_bss<- testData_bss %>% rename("y"="price")
trainData_bss<- trainData %>% rename("Y"="y")
trainData_bss<- trainData_bss %>% rename("y"="price")

library(bestglm)
# BIC + BSS MSE
bg.bic <- bestglm(trainData_bss, family = gaussian, IC="BIC")
bg.bic$BestModel # selected variables : ccutGood, cutVery Good, cutPremium, cutIdeal, depth, x, Y
resp.bic <- predict(bg.bic$BestModel, newdata = testData_bss)
(rmse_bic<-signif(sqrt(mean((testData_bss$y-resp.bic)^2)), 3)) # BIC RMSE

bg.aic <- bestglm(trainData_bss, family = gaussian, IC ="AIC")
bg.aic$BestModel # selected variables : X, carat, cutGood ,cutVery Good ,cutPremium ,cutIdeal ,claritySI1,claritySI2    clarityVS1    clarityVS2   clarityVVS1   clarityVVS2     clarityIF, depth, x, Y
resp.aic <- predict(bg.aic$BestModel, newdata = testData_bss)
(rmse_aic<-signif(sqrt(mean((testData_bss$y-resp.aic)^2)), 3)) # AIC RMSE

plot(resp.bic, testData_bss$y, main = "BIC_Predicted vs Actual Values",
     xlab = "Actual Price", ylab = "Predicted Price", col = "coral", pch = 16)
abline(0, 1, col = "indianred")
plot(resp.aic, testData_bss$y, main = "AIC_Predicted vs Actual Values",
     xlab = "Actual Price", ylab = "Predicted Price", col = "coral3", pch = 16)
abline(0, 1, col = "maroon")

# Forward and Backward Stepwise Selection

# Forward Stepwise Selection
stepwise_model <- step(lm(price ~ 1, data = trainData), 
                       scope = ~ carat + cut + color + clarity + x + y + z + table + depth,
                       direction = "both")

cat("Stepwise Model Summary:\n")
summary(stepwise_model)

# Predictions from the selected stepwise model
predictions_best <- predict(stepwise_model, newdata = testData)

# Plot the predicted vs. actual values
plot(testData$price, predictions_best, main = "Predicted vs Actual Values",
     xlab = "Actual Price", ylab = "Predicted Price", col = "indianred2", pch = 16)
abline(0, 1, col = "brown4")  # Add a 45-degree line for comparison

# MSE for the stepwise model
mse_stepwise <- mean((predictions_best - testData$price)^2)
print(mse_stepwise)

# Calculate Root Mean Squared Error (RMSE)
rmse_stepwise <- signif(sqrt(mse_stepwise), 3)
print(paste("RMSE for the stepwise model:", round(rmse_stepwise, 2)))


### Model Comparison

model_names <- c("Linear regression", "RandomForest", "Decison Tree", "XGBoost", "StepwiseSelection")
rmse_values <- c(rmse, rmse_rf, rmse_tree, rmse_xgb,rmse_stepwise)
model_performance <- data.frame(
  Model = model_names,
  Test_RMSE = rmse_values
)
print(model_performance)

# Print best model
best_model <- model_performance[which.min(model_performance$Test_RMSE), ]
cat("Best model is ", best_model$Model, "and test MSE is ", best_model$Test_RMSE,"\n")

# Plot
ggplot(model_performance, aes(x = Model, y = Test_RMSE, fill = Test_RMSE)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "brown1", high = "brown4") +
  labs(title = "Test RMSE by Model",
       x = "Model",
       y = "Test RMSE") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1) 
  )

# Identifying the best model
best_model_name <- best_model$Model

# Assigning the best model object based on the name
if (best_model_name == "Linear regression") {
  best_model_object <- model_lr
  predictions_best <- predict(best_model_object, newdata = testData)
} else if (best_model_name == "RandomForest") {
  best_model_object <- model_rf
  predictions_best <- predict(best_model_object, newdata = testData)
} else if (best_model_name == "Decison Tree") {
  best_model_object <- model_tree
  predictions_best <- predict(best_model_object, newdata = testData)
} else if (best_model_name == "XGBoost") {
  best_model_object <- model_xgb
  predictions_best <- predict(best_model_object, newdata = test_matrix) # Use matrix for XGBoost
} else if (best_model_name == "StepwiseSelection") {
  best_model_object <- stepwise_model
  predictions_best <- predict(best_model_object, newdata = testData)
}

# Diagnostic Plot
plot(testData$price, predictions_best, main = "Predicted vs Actual Values",
     xlab = "Actual Price", ylab = "Predicted Price", col = "brown2", pch = 16)
abline(0, 1, col = "brown4")  # 45-degree line for comparison


