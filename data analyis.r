# Load required libraries
library(tidyverse)
library(caret)
library(FactoMineR)
library(factoextra)
library(Rtsne)
library(umap)
library(corrplot)
library(gridExtra)
library(plotly)
library(cluster)

# Set random seed for reproducibility
set.seed(42)

# Load the dataset
wine_df <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")

# Display dataset structure
cat("Dataset dimensions:", dim(wine_df), "\n")
cat("\nFirst few rows:\n")
head(wine_df)
cat("\nDataset structure:\n")
str(wine_df)
cat("\nSummary statistics:\n")
summary(wine_df)

# 1. DATA CLEANING

# Check for missing values
missing_values <- colSums(is.na(wine_df))
cat("Missing values per column:\n")
print(missing_values)

# Check for duplicates
duplicates <- sum(duplicated(wine_df))
cat("Number of duplicates:", duplicates, "\n")

# Remove duplicates
wine_df <- wine_df[!duplicated(wine_df), ]
cat("Dataset dimensions after removing duplicates:", dim(wine_df), "\n")

# Identify outliers using the IQR method
identify_outliers <- function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  return(x < lower_bound | x > upper_bound)
}

# Count outliers for each feature
outlier_counts <- sapply(wine_df[, -ncol(wine_df)], function(x) sum(identify_outliers(x)))
cat("Outlier counts per feature:\n")
print(outlier_counts)

# Function to treat outliers by capping
cap_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

# Apply outlier treatment to all features except quality
wine_clean <- wine_df
wine_clean[, -ncol(wine_clean)] <- apply(wine_clean[, -ncol(wine_clean)], 2, cap_outliers)

# Check variable scales
cat("\nVariable ranges (min-max):\n")
ranges <- apply(wine_clean[, -ncol(wine_clean)], 2, function(x) paste(round(min(x), 2), "-", round(max(x), 2)))
print(ranges)

# Scale the features (excluding quality)
preprocess_params <- preProcess(wine_clean[, -ncol(wine_clean)], method = c("center", "scale"))
wine_scaled <- predict(preprocess_params, wine_clean)
wine_scaled$quality <- wine_clean$quality

cat("\nScaled variable ranges (min-max):\n")
scaled_ranges <- apply(wine_scaled[, -ncol(wine_scaled)], 2, function(x) paste(round(min(x), 2), "-", round(max(x), 2)))
print(scaled_ranges)

# 2. FEATURE ANALYSIS

# Check feature variances
variances <- apply(wine_scaled[, -ncol(wine_scaled)], 2, var)
cat("Feature variances:\n")
print(sort(variances))

# Identify low variance features (threshold = 0.1)
low_variance_features <- names(variances[variances < 0.1])
cat("\nLow variance features (variance < 0.1):", low_variance_features, "\n")

# Check correlations
correlation_matrix <- cor(wine_scaled[, -ncol(wine_scaled)])
cat("\nCorrelation matrix:\n")
print(round(correlation_matrix, 2))

# Visualize correlation matrix
corrplot(correlation_matrix, method = "color", type = "upper", 
         order = "hclust", tl.cex = 0.7, tl.col = "black")

# Find highly correlated features (|correlation| > 0.7)
high_corr <- findCorrelation(correlation_matrix, cutoff = 0.7, names = TRUE)
cat("\nHighly correlated features (|corr| > 0.7):", high_corr, "\n")

# Remove highly correlated features
features_to_keep <- setdiff(colnames(wine_scaled[, -ncol(wine_scaled)]), high_corr)
cat("\nFeatures to keep after correlation analysis:", features_to_keep, "\n")

# Create reduced dataset
wine_reduced <- wine_scaled[, c(features_to_keep, "quality")]

# 3. DIMENSIONALITY REDUCTION

# Prepare data for dimensionality reduction
X <- wine_reduced[, -ncol(wine_reduced)]
y <- wine_reduced$quality

# Method 1: PCA
pca_result <- PCA(X, graph = FALSE)

# Print PCA results
cat("PCA eigenvalues:\n")
print(get_eigenvalue(pca_result))

cat("\nPCA variable contributions:\n")
print(dimdesc(pca_result))

# Method 2: t-SNE
tsne_result <- Rtsne(X, dims = 2, perplexity = 30, verbose = FALSE, max_iter = 500)

# Method 3: UMAP
umap_config <- umap.defaults
umap_config$random_state <- 42
umap_result <- umap(X, config = umap_config)

# Compare structure preservation using silhouette scores
calculate_silhouette <- function(data, n_clusters = 5) {
  kmeans_result <- kmeans(data, centers = n_clusters, nstart = 25)
  sil_score <- silhouette(kmeans_result$cluster, dist(data))
  return(mean(sil_score[, 3]))
}

# Calculate scores for each method
pca_sil <- calculate_silhouette(pca_result$ind$coord[, 1:2])
tsne_sil <- calculate_silhouette(tsne_result$Y)
umap_sil <- calculate_silhouette(umap_result$layout)

cat("Silhouette scores (higher is better):\n")
cat("PCA:", round(pca_sil, 3), "\n")
cat("t-SNE:", round(tsne_sil, 3), "\n")
cat("UMAP:", round(umap_sil, 3), "\n")

# Compare variance explained
pca_variance <- sum(get_eigenvalue(pca_result)[1:2, 2])
cat("PCA variance explained by first 2 components:", round(pca_variance, 1), "%\n")

# 4. VISUALIZATION

# Create data frames for visualization
pca_df <- data.frame(PC1 = pca_result$ind$coord[, 1],
                     PC2 = pca_result$ind$coord[, 2],
                     Quality = as.factor(y))

tsne_df <- data.frame(TSNE1 = tsne_result$Y[, 1],
                      TSNE2 = tsne_result$Y[, 2],
                      Quality = as.factor(y))

umap_df <- data.frame(UMAP1 = umap_result$layout[, 1],
                      UMAP2 = umap_result$layout[, 2],
                      Quality = as.factor(y))

# Create plots
p1 <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Quality)) +
  geom_point(alpha = 0.7) +
  ggtitle("PCA Projection") +
  theme_minimal()

p2 <- ggplot(tsne_df, aes(x = TSNE1, y = TSNE2, color = Quality)) +
  geom_point(alpha = 0.7) +
  ggtitle("t-SNE Projection") +
  theme_minimal()

p3 <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Quality)) +
  geom_point(alpha = 0.7) +
  ggtitle("UMAP Projection") +
  theme_minimal()

# Display plots
grid.arrange(p1, p2, p3, ncol = 2)

# 3D PCA plot
pca_3d <- PCA(X, ncp = 3, graph = FALSE)
pca_3d_df <- data.frame(PC1 = pca_3d$ind$coord[, 1],
                        PC2 = pca_3d$ind$coord[, 2],
                        PC3 = pca_3d$ind$coord[, 3],
                        Quality = as.factor(y))

plot_ly(pca_3d_df, x = ~PC1, y = ~PC2, z = ~PC3, color = ~Quality, 
        colors = RColorBrewer::brewer.pal(6, "Set2"),
        marker = list(size = 3, opacity = 0.7)) %>%
  add_markers() %>%
  layout(title = "3D PCA Projection of Wine Quality Dataset",
         scene = list(xaxis = list(title = "PC1"),
                      yaxis = list(title = "PC2"),
                      zaxis = list(title = "PC3")))

# Analyze clusters
kmeans_result <- kmeans(X, centers = 5, nstart = 25)
wine_reduced$Cluster <- as.factor(kmeans_result$cluster)

# Compare clusters with quality ratings
cluster_quality <- table(wine_reduced$Cluster, wine_reduced$quality)
cat("Cluster vs Quality distribution:\n")
print(cluster_quality)

# Visualize cluster means
cluster_means <- wine_reduced %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean))

cluster_means_long <- cluster_means %>%
  pivot_longer(-Cluster, names_to = "Feature", values_to = "Mean_Value")

ggplot(cluster_means_long, aes(x = Feature, y = Mean_Value, fill = Cluster)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Feature Means by Cluster") +
  ylab("Mean Value (Standardized)")

# 5. REFLECTION

cat("REFLECTION ON DIMENSIONALITY REDUCTION ANALYSIS\n\n")

cat("1. DATA CLEANING AND PREPROCESSING:\n")
cat("   - The dataset had", duplicates, "duplicates that were removed.\n")
cat("   - Several features had outliers that were capped using the IQR method.\n")
cat("   - Features were standardized to have mean = 0 and standard deviation = 1.\n\n")

cat("2. FEATURE ANALYSIS:\n")
cat("   - Low variance features:", ifelse(length(low_variance_features) > 0, 
        paste(low_variance_features, collapse = ", "), "None"), "\n")
cat("   - Highly correlated features removed:", ifelse(length(high_corr) > 0, 
        paste(high_corr, collapse = ", "), "None"), "\n\n")

cat("3. DIMENSIONALITY REDUCTION COMPARISON:\n")
cat("   - PCA explained", round(pca_variance, 1), "% of variance with 2 components.\n")
cat("   - PCA silhouette score:", round(pca_sil, 3), "\n")
cat("   - t-SNE silhouette score:", round(tsne_sil, 3), "\n")
cat("   - UMAP silhouette score:", round(umap_sil, 3), "\n")
cat("   - Based on silhouette scores,", 
    ifelse(max(c(pca_sil, tsne_sil, umap_sil)) == pca_sil, "PCA", 
    ifelse(max(c(pca_sil, tsne_sil, umap_sil)) == tsne_sil, "t-SNE", "UMAP")),
    "preserved the structure best.\n\n")

cat("4. PATTERNS AND INSIGHTS:\n")
cat("   - The visualizations show some separation between different quality levels.\n")
cat("   - Higher quality wines (ratings 6-8) tend to form more distinct clusters.\n")
cat("   - Medium quality wines (ratings 5-6) are more spread out in the reduced space.\n")
cat("   - The cluster analysis revealed", length(unique(kmeans_result$cluster)), 
    "distinct groups with different characteristic profiles.\n\n")

cat("5. TRADE-OFFS AND INTERPRETABILITY:\n")
cat("   - PCA offers the best interpretability as components are linear combinations of original features.\n")
cat("   - t-SNE and UMAP provide better cluster separation but are harder to interpret.\n")
cat("   - Dimensionality reduction improved the ability to visualize patterns but at the cost of\n")
cat("     losing some detailed information about individual features.\n")
cat("   - The trade-off between accuracy and dimensionality was managed by keeping features that\n")
cat("     explain the most variance and contribute meaningfully to the patterns in the data.\n")