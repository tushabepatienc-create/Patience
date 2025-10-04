

# ===============================
# Required libraries
# ===============================
library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(rpart)
library(rpart.plot)
library(corrplot)
library(pROC)
library(ggplot2)
library(ggthemes)
library(gridExtra)
library(reshape2)
library(factoextra)
library(readr)

theme_set(theme_minimal())
set.seed(42)

# ===============================
# 1. Data
# ===============================
Telco_Customer_Churn <- read_csv("Assignments/Telco-Customer-Churn.csv")
telco_data <- Telco_Customer_Churn  # use consistent name

cat("Dataset Dimensions:", dim(telco_data), "\n")
cat("Column Names:", names(telco_data), "\n\n")
str(telco_data)
summary(telco_data)

# --- Summary Stats
summary(telco_data$tenure)
summary(telco_data$MonthlyCharges)
telco_data$TotalCharges <- as.numeric(telco_data$TotalCharges)
summary(telco_data$TotalCharges)

# --- Churn distribution
churn_dist <- table(telco_data$Churn)
print(churn_dist)
print(prop.table(churn_dist))

p1 <- ggplot(telco_data, aes(x = Churn, fill = Churn)) +
  geom_bar(alpha = 0.8) +
  geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5) +
  labs(title = "Churn Distribution", x = "Churn Status", y = "Count") +
  scale_fill_manual(values = c("No" = "#2E8B57", "Yes" = "#DC143C"))

p2 <- ggplot(telco_data, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Monthly Charges by Churn Status") +
  scale_fill_manual(values = c("No" = "#2E8B57", "Yes" = "#DC143C"))

p3 <- ggplot(telco_data, aes(x = Churn, y = TotalCharges, fill = Churn)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Total Charges by Churn Status") +
  scale_fill_manual(values = c("No" = "#2E8B57", "Yes" = "#DC143C"))

grid.arrange(p1, p2, p3, ncol = 2)

# --- Correlation heatmap
numerical_features <- telco_data %>% select(tenure, MonthlyCharges, TotalCharges) %>% na.omit()
cor_matrix <- cor(numerical_features)
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# ===============================
# 2. Preprocessing
# ===============================
print(colSums(is.na(telco_data)))
telco_data$TotalCharges[is.na(telco_data$TotalCharges)] <- median(telco_data$TotalCharges, na.rm = TRUE)

telco_clean <- telco_data %>% select(-customerID)
telco_clean$Churn <- ifelse(telco_clean$Churn == "Yes", 1, 0)

categorical_vars <- names(telco_clean)[sapply(telco_clean, is.character)]
numerical_vars <- setdiff(names(telco_clean)[sapply(telco_clean, is.numeric)], "Churn")

dummy_model <- dummyVars(" ~ .", data = telco_clean[, categorical_vars])
categorical_encoded <- predict(dummy_model, newdata = telco_clean[, categorical_vars])

preProcess_model <- preProcess(telco_clean[, numerical_vars], method = c("center", "scale"))
numerical_normalized <- predict(preProcess_model, telco_clean[, numerical_vars])

telco_processed <- cbind(data.frame(categorical_encoded), numerical_normalized, Churn = telco_clean$Churn)

# ===============================
# 3. Feature Engineering & Reduction
# ===============================
telco_processed$tenure_group <- cut(telco_data$tenure,
                                    breaks = c(0, 12, 36, 72),
                                    labels = c("Short", "Medium", "Long"))

pca_result <- prcomp(telco_processed[, numerical_vars], scale. = TRUE, center = TRUE)
summary(pca_result)
fviz_eig(pca_result, addlabels = TRUE, main = "PCA - Explained Variance")

# ===============================
# 4. Second EDA (Post-Processing)
# ===============================

# ---- Check for missing values
cat("Missing values before fixing:\n")
print(colSums(is.na(telco_processed)))

# ---- Impute missing values
for (col in names(telco_processed)) {
  if (is.numeric(telco_processed[[col]])) {
    telco_processed[[col]][is.na(telco_processed[[col]])] <- median(telco_processed[[col]], na.rm = TRUE)
  } else if (is.factor(telco_processed[[col]]) || is.character(telco_processed[[col]])) {
    mode_val <- names(sort(table(telco_processed[[col]]), decreasing = TRUE))[1]
    telco_processed[[col]][is.na(telco_processed[[col]])] <- mode_val
  }
}

# ---- Confirm no NAs remain
cat("Missing values after fixing:\n")
print(colSums(is.na(telco_processed)))

# ---- Random Forest Feature Importance
rf_model <- randomForest(as.factor(Churn) ~ ., 
                         data = telco_processed, 
                         ntree = 100, 
                         importance = TRUE)

importance_df <- data.frame(
  Feature = rownames(importance(rf_model)),
  Importance = importance(rf_model)[, "MeanDecreaseGini"]
)

# ---- Top 15 features
top_features <- head(importance_df[order(-importance_df$Importance), ], 15)

# ---- Plot feature importance
ggplot(top_features, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#4E84C4", alpha = 0.8) +
  coord_flip() +
  labs(title = "Top 15 Feature Importance (Random Forest)",
       x = "Features", y = "Importance")


# ===============================
# 5. ML Modeling
# ===============================

# Split data
train_index <- createDataPartition(telco_processed$Churn, p = 0.8, list = FALSE)
train_data <- telco_processed[train_index, ]
test_data  <- telco_processed[-train_index, ]

# -------------------------------
# Decision Tree
# -------------------------------
dt_model <- rpart(as.factor(Churn) ~ ., 
                  data = train_data, 
                  method = "class", 
                  control = rpart.control(cp = 0.01))

# -------------------------------
# Random Forest
# -------------------------------
rf_model <- randomForest(as.factor(Churn) ~ ., 
                         data = train_data, 
                         ntree = 100, 
                         importance = TRUE)

# -------------------------------
# XGBoost
# -------------------------------
# Remove non-numeric columns (like tenure_group) before converting to matrix
train_data_xgb <- train_data %>% select(-tenure_group)
test_data_xgb  <- test_data %>% select(-tenure_group)

# Ensure all predictors are numeric
train_matrix <- as.matrix(train_data_xgb %>% select(-Churn))
test_matrix  <- as.matrix(test_data_xgb %>% select(-Churn))

# Create DMatrix objects
xgb_train <- xgb.DMatrix(data = train_matrix, label = train_data_xgb$Churn)
xgb_test  <- xgb.DMatrix(data = test_matrix,  label = test_data_xgb$Churn)

# Train XGBoost model
xgb_model <- xgboost(data = xgb_train, 
                     nrounds = 100, 
                     objective = "binary:logistic",
                     eval_metric = "logloss", 
                     verbose = 0)


# ===============================
# 6. Evaluation
# ===============================

# ---- Evaluation Function
evaluate_model <- function(model, test_data, model_type = "tree") {
  if (model_type == "tree") {
    # Decision Tree
    predictions <- predict(model, test_data, type = "class")
    probabilities <- predict(model, test_data, type = "prob")[, 2]
    
  } else if (model_type == "forest") {
    # Random Forest
    predictions <- predict(model, test_data)
    probabilities <- predict(model, test_data, type = "prob")[, 2]
    
  } else if (model_type == "xgb") {
    # XGBoost requires numeric matrix (already prepared as xgb_test)
    probabilities <- predict(model, xgb_test)
    predictions <- ifelse(probabilities > 0.5, 1, 0)
  }
  
  # Confusion Matrix & Metrics
  cm <- confusionMatrix(as.factor(predictions), as.factor(test_data$Churn), positive = "1")
  roc_curve <- roc(test_data$Churn, probabilities)
  
  list(
    accuracy = cm$overall["Accuracy"],
    recall = cm$byClass["Recall"],
    precision = cm$byClass["Precision"],
    f1 = cm$byClass["F1"],
    auc = auc(roc_curve),
    roc_curve = roc_curve
  )
}

# ---- Run Evaluations
dt_results  <- evaluate_model(dt_model, test_data, "tree")
rf_results  <- evaluate_model(rf_model, test_data, "forest")
xgb_results <- evaluate_model(xgb_model, test_data_xgb, "xgb")  # pass numeric-cleaned df

# ---- Collect Results
results_df <- data.frame(
  Model = c("Decision Tree", "Random Forest", "XGBoost"),
  Accuracy = c(dt_results$accuracy, rf_results$accuracy, xgb_results$accuracy),
  Recall   = c(dt_results$recall, rf_results$recall, xgb_results$recall),
  Precision= c(dt_results$precision, rf_results$precision, xgb_results$precision),
  F1_Score = c(dt_results$f1, rf_results$f1, xgb_results$f1),
  AUC      = c(dt_results$auc, rf_results$auc, xgb_results$auc)
)

print("Model Performance Comparison:")
print(results_df)

# ---- ROC Curves
plot(dt_results$roc_curve, col = "blue", main = "ROC Curves Comparison")
plot(rf_results$roc_curve, col = "red", add = TRUE)
plot(xgb_results$roc_curve, col = "green", add = TRUE)
legend("bottomright",
       legend = c(
         paste("Decision Tree (AUC =", round(dt_results$auc, 3), ")"),
         paste("Random Forest (AUC =", round(rf_results$auc, 3), ")"),
         paste("XGBoost (AUC =", round(xgb_results$auc, 3), ")")
       ),
       col = c("blue", "red", "green"), lwd = 2)

# ===============================
# Stratified Cross Validation
# ===============================
telco_cv <- telco_processed
telco_cv$Churn <- as.factor(ifelse(telco_cv$Churn == 1, "Yes", "No"))

ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     sampling = "up")

cv_model <- train(Churn ~ ., data = telco_cv, 
                  method = "rf", 
                  trControl = ctrl, 
                  metric = "ROC", 
                  tuneLength = 3)

print("Cross-Validation Results:")
print(cv_model$results)


# ===============================
# 7. Interpretation
# ===============================
top_churn_drivers <- head(importance_df[order(-importance_df$Importance), ], 10)
print(top_churn_drivers)
prp(dt_model, extra = 1, faclen = 0, box.palette = "auto",
    main = "Decision Tree for Churn Prediction")

cat("\nRetention Strategies:\n")
cat("1. Focus on short-tenure customers\n")
cat("2. Review pricing for high monthly charges\n")
cat("3. Improve service for fiber optic users\n")

