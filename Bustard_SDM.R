###############################################################################
# CODE FOR: Great Bustard Habitat Suitability Modeling
# AUTHORS: Gankhuyag Purev-Ochir1,2*, Hussain Saifee Reshamwala2,3, Amarkhuu Gungaa2, Baasansuren Erdenechimeg2, Tsoggerel Baldandugar2, Yves Hingrat4 & Guo Yumin1*

1School of Ecology and Nature Conservation, Beijing Forestry University, Beijing, China 2Mongolian Bird Conservation Center, Sukhbaatar District, Ulaanbaatar, Mongolia 
3 Institute of Environmental Sciences, Jagiellonian University, 7 Gronostajowa str., 30-387 Krakow, Poland
4Reneco International Wildlife Consultants LTD, Abu Dhabi, United Arab Emirates
*Correspondence: bird168@126.com; pgankhuyag@gmail.com; 

# JOURNAL: PLOS ONE
#
# DESCRIPTION:
# This script performs ensemble species distribution modeling (SDM) for the 
# Great Bustard (Otis tarda) using 9 environmental layers and 12 algorithms.
# The workflow includes:
#   1. Data preparation and spatial blocking for cross-validation
#   2. Model training with multiple algorithms (RF, GBM, GLM, MARS, XGBOOST,
#      SVM_radial, FDA, MAXENT, ANN, CTA, SRE)
#   3. Spatial cross-validation with 5 folds
#   4. Model performance evaluation (TSS, AUC)
#   5. Ensemble prediction creation
#   6. Variable importance analysis
#   7. Response curve generation
#   8. Future climate projections (2050, 2070, 2090 under SSP1)
#   9. Threshold optimization
#   10. Climate novelty analysis
#
# REQUIREMENTS:
# - R version: 4.2.1 or higher
# - Dependencies: blockCV, randomForest, ranger, gbm, xgboost, dismo, raster,
#   caret, MASS, e1071, earth, maxnet, nnet, mda, kernlab, ggplot2, patchwork,
#   viridis, pROC, reshape2, tidyr, GGally, spdep, sp
#
# INPUT FILES (USER MUST UPDATE PATHS):
# - Occurrence data: Final_GB_locs_correct_hist (data frame with lon/lat)
# - Environmental layers (current): aligned_bio1.tif, aligned_bio2.tif, 
#   aligned_bio3.tif, aligned_bio5.tif, aligned_bio7.tif, aligned_bio12.tif,
#   Water_dist_mask.tif, aligned_HF.tif, aligned_elv.tif
# - Future layers (2050): aligned_2050Bioclim_Var_*.tif
# - Future layers (2070): aligned_2070bio*.tif
# - Future layers (2090): aligned_2090bio*.tif
#
# OUTPUT FILES:
# - performance.csv: Model performance metrics
# - TSS_vs_AUC_Biomod2_Style.png: Performance visualization
# - Ensemble_SDM_Prediction.tif: Current ensemble habitat suitability map
# - Ensemble_Future_SSP1_*x.tif: Future ensemble predictions
# - Variable_Importance_*.csv and .png: Variable importance analyses
# - Response_Curves_Faceted.png: Response curves for all variables
# - Threshold_Decision_Plot.png: Threshold optimization plot
# - Novel_Climate_Areas.tif: Areas with novel future climate conditions
# - Current_vs_Future_Layer_Comparison.csv: Climate layer comparisons
#
# LAST UPDATED: [Insert Date]
###############################################################################

# =============================================================================
# SECTION 1: SET WORKING DIRECTORY AND LOAD LIBRARIES
# =============================================================================

# Set working directory to location of data files
# USER MUST UPDATE THIS PATH TO THEIR DATA DIRECTORY
setwd("E:/50 km Block")

# Load all required libraries
# These packages are essential for species distribution modeling, spatial analysis,
# machine learning, and visualization
library(blockCV)          # Spatial cross-validation with blocking
library(randomForest)     # Random Forest algorithm
library(ranger)           # Fast Random Forest implementation
library(gbm)              # Gradient Boosting Machine
library(xgboost)          # Extreme Gradient Boosting
library(dismo)            # Species distribution modeling (bioclim, evaluate)
library(raster)           # Raster data manipulation
library(caret)            # Classification and regression training
library(MASS)             # Statistical functions (for GLM)
library(e1071)            # Support Vector Machines
library(earth)            # Multivariate Adaptive Regression Splines (MARS)
library(maxnet)           # MaxEnt algorithm
library(nnet)             # Neural Networks
library(mda)              # Flexible Discriminant Analysis (FDA)
library(kernlab)          # Kernel-based machine learning
library(ggplot2)          # Data visualization
library(patchwork)        # Combining multiple ggplot plots
library(viridis)          # Colorblind-friendly color palettes
library(pROC)             # ROC curve analysis
library(reshape2)         # Data reshaping

# =============================================================================
# SECTION 2: LOAD AND PREPARE OCCURRENCE DATA
# =============================================================================

# Step 1: Load and prepare occurrence data
# 'Final_GB_locs_correct_hist' should be a data frame containing presence locations
# with columns "lon" (longitude) and "lat" (latitude)

# Extract longitude and latitude columns from occurrence data
occ <- Final_GB_locs_correct_hist[, c("lon", "lat")]

# Remove rows with missing coordinates (NA values)
occ <- occ[complete.cases(occ), ]

# Remove duplicate occurrence records (same coordinates)
occ <- occ[!duplicated(occ), ]

# Print number of unique occurrence points to console
cat("Number of occurrence points:", nrow(occ), "\n")

# =============================================================================
# SECTION 3: LOAD ENVIRONMENTAL LAYERS
# =============================================================================

# Step 2: Load the 9 environmental layers used for modeling
# These layers were pre-processed and aligned to the same extent, resolution,
# and coordinate reference system

# USER MUST UPDATE THESE PATHS TO THEIR ACTUAL FILE LOCATIONS

# Bioclimatic variables (WorldClim)
bio01 <- raster("D:/Bustard Project/TRY2 with 12 models/Layers for analysis/Aligned/aligned_bio1.tif")  # Annual Mean Temperature
bio02 <- raster("D:/Bustard Project/TRY2 with 12 models/Layers for analysis/Aligned/aligned_bio2.tif")  # Mean Diurnal Range
bio03 <- raster("D:/Bustard Project/TRY2 with 12 models/Layers for analysis/Aligned/aligned_bio3.tif")  # Isothermality
bio05 <- raster("D:/Bustard Project/TRY2 with 12 models/Layers for analysis/Aligned/aligned_bio5.tif")  # Max Temperature of Warmest Month
bio07 <- raster("D:/Bustard Project/TRY2 with 12 models/Layers for analysis/Aligned/aligned_bio7.tif")  # Temperature Annual Range
bio12 <- raster("D:/Bustard Project/TRY2 with 12 models/Layers for analysis/Aligned/aligned_bio12.tif") # Annual Precipitation

# Anthropogenic and topographic layers
water_dist <- raster("D:/Bustard Project/TRY2 with 12 models/Layers for analysis/Water_dist_mask.tif") # Distance to water
HF <- raster("D:/Bustard Project/TRY2 with 12 models/Layers for analysis/Aligned/aligned_HF.tif")    # Human Footprint index
elv <- raster("D:/Bustard Project/TRY2 with 12 models/Layers for analysis/Aligned/aligned_elv.tif")  # Elevation

# Create a raster stack combining all 9 layers
# A raster stack allows efficient processing of multiple layers together
env_stack <- stack(bio01, bio02, bio03, bio05, bio07, bio12, water_dist, HF, elv)

# Assign meaningful names to each layer for easier reference
names(env_stack) <- c("bio01", "bio02", "bio03", "bio05", "bio07", "bio12", 
                      "water_dist", "HF", "elv")

# Print the layer names to confirm they are correctly loaded
cat("Your 9 fixed environmental layers:\n")
print(names(env_stack))

# Visualize the environmental layers (opens a plot window)
plot(env_stack)

# =============================================================================
# SECTION 4: CREATE SPATIAL BLOCKS FOR CROSS-VALIDATION
# =============================================================================

# Step 3: Create spatial blocks for cross-validation
# Spatial blocking reduces autocorrelation between training and test sets
# by ensuring spatially separated folds

cat("Creating spatial blocks...\n")

# Convert occurrence points to SpatialPoints object with WGS84 projection
occ_sp <- SpatialPoints(occ, proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs"))

# Create spatial CV folds using blockCV package
# - size: block size in meters (50,000m = 50km blocks)
# - k: number of folds (5-fold cross-validation)
# - selection: "random" selection of blocks to folds
# - iteration: number of random iterations for block assignment
sb <- cv_spatial(x = occ_sp,
                 column = NULL,  # No species column needed
                 r = env_stack,
                 size = 50000,   # Block size in meters (50 km)
                 k = 5,
                 selection = "random",
                 iteration = 50,
                 plot = FALSE)

# Extract fold IDs for each occurrence point
fold_ids <- sb$folds_ids

# Print distribution of points across folds
cat("Spatial blocks created. Points per block:\n")
print(table(fold_ids))

# =============================================================================
# SECTION 5: VISUALIZE SPATIAL BLOCKS
# =============================================================================

# Visualize the spatial blocks to check if they are well-distributed
cat("\nVisualizing spatial blocks...\n")

# Convert spatial points to data frame for plotting with ggplot2
occ_df <- as.data.frame(occ_sp)
colnames(occ_df) <- c("lon", "lat")

# Create data frame with fold assignments for plotting
fold_df <- data.frame(
  lon = occ_df$lon,
  lat = occ_df$lat,
  Fold = as.factor(fold_ids)
)

# Plot 1: Points colored by fold assignment
# This shows which points belong to which spatial fold
p1 <- ggplot() +
  geom_point(data = fold_df, aes(x = lon, y = lat, color = Fold), size = 2) +
  scale_color_viridis_d() +
  labs(title = "Spatial Blocks: Points Colored by Fold",
       subtitle = paste("Total points:", nrow(occ_df), "| Folds:", length(unique(fold_ids))),
       x = "Longitude", y = "Latitude") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "right")

# Plot 2: Density plot showing point distribution with contours
# This helps visualize the spatial clustering of points
p2 <- ggplot(fold_df, aes(x = lon, y = lat)) +
  geom_point(aes(color = Fold), alpha = 0.6, size = 1.5) +
  geom_density_2d(color = "black", alpha = 0.5) +
  scale_color_viridis_d() +
  labs(title = "Spatial Blocks with Density Contours",
       x = "Longitude", y = "Latitude") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

# Plot 3: Bar plot showing number of points per fold
# Ensures each fold has sufficient points for model training
fold_counts <- as.data.frame(table(fold_ids))
colnames(fold_counts) <- c("Fold", "Count")

p3 <- ggplot(fold_counts, aes(x = Fold, y = Count, fill = Fold)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  geom_text(aes(label = Count), vjust = -0.5) +
  scale_fill_viridis_d() +
  labs(title = "Points per Spatial Block",
       x = "Fold", y = "Number of Points") +
  ylim(0, max(fold_counts$Count) * 1.1) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

# Combine the three plots into a single figure using patchwork
block_plots <- (p1 | p2) / p3 +
  plot_annotation(
    title = "Spatial Cross-Validation Block Visualization",
    subtitle = paste("Block size: 50,000m | Number of folds:", length(unique(fold_ids))),
    caption = paste("Generated", format(Sys.Date(), "%B %d, %Y")),
    theme = theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )
  )

# Display the combined plot
block_plots

# Save the spatial blocks visualization as a high-resolution PNG file
ggsave("Spatial_Blocks_Visualization.png", block_plots, width = 12, height = 10, dpi = 300)
cat("✓ Spatial blocks visualization saved: Spatial_Blocks_Visualization.png\n")

# =============================================================================
# SECTION 6: PREPARE MODELING DATA (PRESENCE AND BACKGROUND POINTS)
# =============================================================================

# Step 4: Prepare modeling data by extracting environmental values
# at presence locations and generating background (pseudo-absence) points

cat("Preparing data for modeling...\n")

# Extract environmental values at presence points
# This creates a data frame with PresAbs = 1 (presence) and the environmental values
presence_data <- data.frame(
  PresAbs = 1,
  raster::extract(env_stack, occ)
)

# Remove any rows with missing data (NA values)
presence_data <- presence_data[complete.cases(presence_data), ]

# Generate background (pseudo-absence) points
# Using 3 times as many background points as presence points is a common practice
set.seed(123)  # Set seed for reproducibility
bg_points <- randomPoints(env_stack, n = nrow(presence_data) * 3)

# Extract environmental values at background points
absence_data <- data.frame(
  PresAbs = 0,
  raster::extract(env_stack, bg_points)
)

# Remove rows with missing data
absence_data <- absence_data[complete.cases(absence_data), ]

# Combine presence and absence data into a single modeling dataset
model_data <- rbind(presence_data, absence_data)
model_data <- model_data[complete.cases(model_data), ]

# Print dataset summary
cat("Modeling dataset:", nrow(model_data), "rows\n")
cat("  - Presences:", sum(model_data$PresAbs == 1), "\n")
cat("  - Absences:", sum(model_data$PresAbs == 0), "\n")

# =============================================================================
# SECTION 7: VISUALIZE PRESENCE AND BACKGROUND POINTS
# =============================================================================

# Simple visualization of presence and background point locations
# This helps verify that background points cover the study area appropriately
plot_data <- rbind(
  data.frame(lon = occ$lon, lat = occ$lat, Type = "Presence"),
  data.frame(lon = bg_points[,1], lat = bg_points[,2], Type = "Background")
)

simple_plot <- ggplot(plot_data, aes(x = lon, y = lat, color = Type)) +
  geom_point(size = 1.5, alpha = 0.7) +
  scale_color_manual(values = c("Presence" = "red", "Background" = "blue")) +
  labs(title = "Presence and Background Points",
       subtitle = paste("Presence:", sum(model_data$PresAbs == 1), 
                        " | Background:", sum(model_data$PresAbs == 0),
                        " | Ratio:", round(sum(model_data$PresAbs == 0)/sum(model_data$PresAbs == 1), 1), ":1"),
       x = "Longitude", y = "Latitude") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

# Display and save the plot
print(simple_plot)
ggsave("Presence_Background_Simple.png", simple_plot, width = 8, height = 6, dpi = 300)
cat("✓ Simple presence-background plot saved: Presence_Background_Simple.png\n")

# =============================================================================
# SECTION 8: DENSITY PLOTS FOR ENVIRONMENTAL VARIABLES
# =============================================================================

# Create density plots comparing presence vs background distributions
# for each environmental variable. This helps identify which variables
# differentiate suitable from unsuitable habitat

cat("\nCreating density plots for environmental variables...\n")

# Create a list to store individual density plots
density_plots <- list()

# Get all environmental variable names (excluding the PresAbs column)
env_vars <- names(model_data)[-1]

# Create a density plot for each environmental variable
for(var_name in env_vars) {
  # Create data frame for this variable with presence/background labels
  var_data <- data.frame(
    Value = model_data[[var_name]],
    Type = ifelse(model_data$PresAbs == 1, "Presence", "Background")
  )
  
  # Create density plot with overlaid distributions
  p <- ggplot(var_data, aes(x = Value, fill = Type, color = Type)) +
    geom_density(alpha = 0.5, size = 0.8) +
    scale_fill_manual(values = c("Presence" = "#E15759", "Background" = "#4E79A7")) +
    scale_color_manual(values = c("Presence" = "#E15759", "Background" = "#4E79A7")) +
    labs(
      title = paste("Distribution:", var_name),
      x = var_name,
      y = "Density"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 10, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 9),
      legend.position = "none",
      panel.grid.minor = element_blank()
    )
  
  # Store the plot in the list
  density_plots[[var_name]] <- p
}

# Combine all density plots into a 3x3 grid (for 9 variables)
combined_density_plot <- wrap_plots(density_plots, ncol = 3, nrow = 3) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

# Add overall title and subtitle
combined_density_plot <- combined_density_plot +
  plot_annotation(
    title = "Environmental Variable Distributions: Presence vs Background",
    subtitle = paste("Comparing", nrow(presence_data), "presence vs", nrow(absence_data), "background points"),
    caption = paste("Generated", format(Sys.Date(), "%Y-%m-%d")),
    theme = theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 10)
    )
  )

# Display and save the combined density plot
combined_density_plot
ggsave("Environmental_Variables_Density.png", combined_density_plot, 
       width = 14, height = 12, dpi = 300)
cat("✓ Density plots saved: Environmental_Variables_Density.png\n")

# =============================================================================
# SECTION 9: SPATIAL CROSS-VALIDATION FUNCTION
# =============================================================================

# Step 5: Define the spatial cross-validation evaluation function
# This function trains and evaluates multiple algorithms using spatial folds
# to ensure unbiased performance estimates

spatial_cv_evaluation <- function(model_data, fold_ids) {
  
  # Define the algorithms to test
  # Removed duplicates (FDA appears twice in original list)
  methods <- c("RF", "GBM", "GLM", "MARS", "XGBOOST",
               "SVM_radial", "FDA", "MAXENT", 
               "ANN", "CTA", "SRE")
  
  # Results storage containers
  all_results <- list()       # Performance metrics per fold per method
  all_predictions <- list()   # Predicted values per fold per method
  all_models <- list()        # Trained models per fold per method
  
  # Loop through each algorithm/method
  for(method in methods) {
    cat("Processing", method, "...\n")
    method_results <- data.frame()
    method_predictions <- list()
    method_models <- list()
    
    # Loop through each spatial fold for cross-validation
    for(fold in 1:max(fold_ids)) {
      # Split data using spatial block assignments
      # Points in the current fold are used for testing; all others for training
      test_indices <- which(fold_ids == fold)
      train_indices <- which(fold_ids != fold)
      
      # Create training and test datasets
      # Note: model_data is structured with presences first, then absences
      # So we need to select indices from both presence and absence sections
      train_data <- model_data[c(train_indices, train_indices + nrow(presence_data)), ]
      test_data <- model_data[c(test_indices, test_indices + nrow(presence_data)), ]
      
      # Remove any rows with missing data
      train_data <- train_data[complete.cases(train_data), ]
      test_data <- test_data[complete.cases(test_data), ]
      
      # Check if we have enough data for training and testing
      if(nrow(train_data) > 10 && nrow(test_data) > 5) {
        
        # Try to train the model for this fold; skip if an error occurs
        tryCatch({
          # ===========================================
          # RANDOM FOREST (RF)
          # ===========================================
          if(method == "RF") {
            model <- randomForest::randomForest(
              as.factor(PresAbs) ~ .,
              data = train_data,
              ntree = 500,
              importance = TRUE
            )
            pred_probs <- predict(model, test_data, type = "prob")[,2]
            
            # ===========================================
            # GRADIENT BOOSTING MACHINE (GBM)
            # ===========================================
          } else if(method == "GBM") {
            model <- gbm::gbm(
              PresAbs ~ .,
              data = train_data,
              distribution = "bernoulli",
              n.trees = 1000,
              interaction.depth = 3,
              shrinkage = 0.01,
              cv.folds = 5
            )
            best_trees <- gbm::gbm.perf(model, plot.it = FALSE)
            pred_probs <- predict(model, test_data, n.trees = best_trees, type = "response")
            
            # ===========================================
            # GENERALIZED LINEAR MODEL (GLM)
            # ===========================================
          } else if(method == "GLM") {
            model <- glm(PresAbs ~ .,
                         data = train_data,
                         family = binomial)
            pred_probs <- predict(model, test_data, type = "response")
            
            # ===========================================
            # MULTIVARIATE ADAPTIVE REGRESSION SPLINES (MARS)
            # ===========================================
          } else if(method == "MARS") {
            model <- earth::earth(
              PresAbs ~ .,
              data = train_data,
              glm = list(family = binomial)
            )
            pred_probs <- predict(model, test_data, type = "response")
            
            # ===========================================
            # EXTREME GRADIENT BOOSTING (XGBOOST)
            # ===========================================
          } else if(method == "XGBOOST") {
            train_matrix <- xgboost::xgb.DMatrix(
              data = as.matrix(train_data[, -1]), 
              label = train_data$PresAbs
            )
            test_matrix <- xgboost::xgb.DMatrix(
              data = as.matrix(test_data[, -1])
            )
            
            model <- xgboost::xgboost(
              data = train_matrix,
              nrounds = 100,
              objective = "binary:logistic",
              eval_metric = "logloss",
              verbose = 0
            )
            pred_probs <- predict(model, test_matrix)
            
            # ===========================================
            # SUPPORT VECTOR MACHINE (SVM) - Radial Kernel
            # ===========================================
          } else if(method == "SVM_radial") {
            model <- e1071::svm(
              as.factor(PresAbs) ~ .,
              data = train_data,
              probability = TRUE,
              kernel = "radial"
            )
            pred <- predict(model, test_data, probability = TRUE)
            pred_probs <- attr(pred, "probabilities")[,2]
            
            # ===========================================
            # FLEXIBLE DISCRIMINANT ANALYSIS (FDA)
            # ===========================================
          } else if(method == "FDA") {
            model <- mda::fda(
              as.factor(PresAbs) ~ .,
              data = train_data
            )
            pred_probs <- predict(model, test_data, type = "posterior")[,2]
            
            # ===========================================
            # MAXIMUM ENTROPY (MAXENT)
            # ===========================================
          } else if(method == "MAXENT") {
            model <- maxnet::maxnet(
              p = train_data$PresAbs,
              data = train_data[, -1]
            )
            pred_probs <- predict(model, test_data[, -1], type = "logistic")
            
            # ===========================================
            # ARTIFICIAL NEURAL NETWORK (ANN)
            # ===========================================
          } else if(method == "ANN") {
            model <- nnet::nnet(
              as.factor(PresAbs) ~ .,
              data = train_data,
              size = 5,
              decay = 0.1,
              maxit = 200,
              trace = FALSE,
              linout = FALSE
            )
            pred_probs <- predict(model, test_data, type = "raw")
            
            # ===========================================
            # CLASSIFICATION TREE ANALYSIS (CTA)
            # ===========================================
          } else if(method == "CTA") {
            model <- rpart::rpart(
              as.factor(PresAbs) ~ .,
              data = train_data,
              method = "class",
              control = rpart::rpart.control(cp = 0.01, minsplit = 10)
            )
            pred_probs <- predict(model, test_data, type = "prob")[,2]
            
            # ===========================================
            # SURFACE RANGE ENVELOPE (SRE) - Bioclim
            # ===========================================
          } else if(method == "SRE") {
            model <- dismo::bioclim(
              x = train_data[train_data$PresAbs == 1, -1]
            )
            pred_probs <- predict(model, test_data[, -1])
          }
          
          # Calculate evaluation metrics if we have valid predictions
          if(length(unique(test_data$PresAbs)) > 1 && 
             length(pred_probs) == nrow(test_data)) {
            
            # Separate predictions for presences and absences
            pres_preds <- pred_probs[test_data$PresAbs == 1]
            abs_preds <- pred_probs[test_data$PresAbs == 0]
            
            if(length(pres_preds) > 0 && length(abs_preds) > 0) {
              # Calculate TSS (True Skill Statistic) and AUC (Area Under Curve)
              eval <- dismo::evaluate(p = pres_preds,
                                      a = abs_preds)
              
              tss <- eval@TPR + eval@TNR - 1  # TSS = Sensitivity + Specificity - 1
              auc <- eval@auc
              
              # Store results for this fold
              method_results <- rbind(method_results, data.frame(
                Fold = fold,
                Method = method,
                TSS = tss,
                AUC = auc,
                Train_Size = nrow(train_data),
                Test_Size = nrow(test_data)
              ))
              
              # Store predictions for ROC curve generation
              method_predictions[[fold]] <- data.frame(
                Fold = fold,
                Method = method,
                Observed = test_data$PresAbs,
                Predicted = pred_probs
              )
              
              # Store the trained model
              method_models[[fold]] <- model
            }
          }
          
        }, error = function(e) {
          # Print error message but continue with next fold
          cat("Error in", method, "fold", fold, ":", e$message, "\n")
        })
      }
    }
    
    # Store results for this method if any folds were successful
    if(nrow(method_results) > 0) {
      all_results[[method]] <- method_results
      if(length(method_predictions) > 0) {
        all_predictions[[method]] <- do.call(rbind, method_predictions)
      }
      all_models[[method]] <- method_models
      cat("  -", method, "completed with", nrow(method_results), "folds\n")
    } else {
      cat("  -", method, "failed on all folds\n")
    }
  }
  
  # Return all results as a list
  return(list(results = all_results, predictions = all_predictions, models = all_models))
}

# =============================================================================
# SECTION 10: RUN SPATIAL CROSS-VALIDATION
# =============================================================================

# Step 6: Execute spatial cross-validation for all algorithms
cat("Running spatial cross-validation for selected algorithms...\n")
cv_results <- spatial_cv_evaluation(model_data, fold_ids)

# =============================================================================
# SECTION 11: PERFORMANCE ANALYSIS AND VISUALIZATION
# =============================================================================

# Step 7: Analyze and visualize model performance
cat("Analyzing model performance...\n")

# Combine all results from all methods and folds
all_performance <- do.call(rbind, cv_results$results)
all_performance <- all_performance[complete.cases(all_performance), ]

# Calculate mean performance per method (TSS and AUC)
performance_summary <- aggregate(cbind(TSS, AUC) ~ Method, 
                                 data = all_performance, 
                                 FUN = function(x) c(Mean = mean(x), SD = sd(x)))

# Extract mean and standard deviation for each metric
performance_summary$Mean_TSS <- performance_summary$TSS[, "Mean"]
performance_summary$SD_TSS <- performance_summary$TSS[, "SD"]
performance_summary$Mean_AUC <- performance_summary$AUC[, "Mean"]
performance_summary$SD_AUC <- performance_summary$AUC[, "SD"]
performance_summary <- performance_summary[, c("Method", "Mean_TSS", "SD_TSS", "Mean_AUC", "SD_AUC")]

# Sort by TSS (best performing models first)
performance_summary <- performance_summary[order(-performance_summary$Mean_TSS), ]

# Print performance ranking
cat("MODEL PERFORMANCE RANKING:\n")
print(performance_summary)

# Save performance summary to CSV
write.csv(performance_summary, "performance.csv", row.names = FALSE)

# =============================================================================
# SECTION 12: BIOMOD2-STYLE TSS VS AUC PLOT
# =============================================================================

# Step 8: Create a biomod2-style performance plot (TSS vs AUC)
cat("\nCreating Biomod2-style TSS vs AUC plot...\n")

# Create the plot with error bars showing standard deviation
biomod2_style_plot <- ggplot(performance_summary, aes(x = Mean_AUC, y = Mean_TSS)) +
  # Points colored by method, sized by TSS value
  geom_point(aes(color = Method, size = Mean_TSS), alpha = 0.8) +
  # Vertical error bars (TSS ± SD)
  geom_errorbar(aes(ymin = Mean_TSS - SD_TSS, ymax = Mean_TSS + SD_TSS), 
                width = 0.005, alpha = 0.5) +
  # Horizontal error bars (AUC ± SD)
  geom_errorbarh(aes(xmin = Mean_AUC - SD_AUC, xmax = Mean_AUC + SD_AUC), 
                 height = 0.01, alpha = 0.5) +
  # Labels for each method
  geom_text(aes(label = Method), hjust = -0.1, vjust = 0.5, size = 3) +
  scale_color_viridis_d() +
  scale_size_continuous(range = c(3, 8)) +
  labs(title = "Model Performance: TSS vs AUC",
       subtitle = "Similar to biomod2 evaluation plot",
       x = "Area Under Curve (AUC) ± SD",
       y = "True Skill Statistic (TSS) ± SD",
       caption = "Point size represents TSS value") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
        plot.subtitle = element_text(hjust = 0.5),
        panel.grid.major = element_line(color = "gray90"),
        panel.grid.minor = element_line(color = "gray95")) +
  # Add reference lines for performance thresholds
  geom_hline(yintercept = c(0.2, 0.4, 0.6, 0.8), linetype = "dashed", alpha = 0.2) +
  geom_vline(xintercept = c(0.6, 0.7, 0.8, 0.9), linetype = "dashed", alpha = 0.2) +
  # Highlight "good performance" region
  annotate("rect", xmin = 0.7, xmax = 1.0, ymin = 0.4, ymax = 1.0,
           alpha = 0.1, fill = "green") +
  annotate("text", x = 0.85, y = 0.45, label = "Good Performance",
           color = "darkgreen", size = 3)

# Save the plot
ggsave("TSS_vs_AUC_Biomod2_Style.png", biomod2_style_plot, width = 10, height = 8, dpi = 300)

# Display the plot
biomod2_style_plot

# Save the data used for the plot
tss_auc_data <- performance_summary[, c("Method", "Mean_TSS", "SD_TSS", "Mean_AUC", "SD_AUC")]
write.csv(tss_auc_data, "TSS_AUC_Plot_Data.csv", row.names = FALSE)

cat("✓ Biomod2-style plot saved: TSS_vs_AUC_Biomod2_Style.png\n")
cat("✓ Plot data saved: TSS_AUC_Plot_Data.csv\n")

# =============================================================================
# SECTION 13: TRADITIONAL PERFORMANCE COMPARISON BOXPLOTS
# =============================================================================

# Step 9: Create traditional boxplots comparing TSS and AUC across methods
p1 <- ggplot(all_performance, aes(x = Method, y = TSS, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  scale_fill_viridis_d() +
  labs(title = "Model Performance Comparison (TSS)",
       subtitle = paste(length(unique(all_performance$Method)), "Algorithms"),
       y = "True Skill Statistic (TSS)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

p2 <- ggplot(all_performance, aes(x = Method, y = AUC, fill = Method)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  scale_fill_viridis_d() +
  labs(title = "Model Performance Comparison (AUC)", 
       y = "Area Under Curve (AUC)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Combine performance plots vertically
performance_plot <- p1 / p2
ggsave("Model_Performance_Comparison.png", performance_plot, width = 12, height = 10, dpi = 300)

performance_plot

# =============================================================================
# SECTION 14: SELECT TOP MODELS FOR ENSEMBLE
# =============================================================================

# Step 10: Select top models for ensemble creation
# Use top 3 models (or fewer if less than 3 are available)
top_methods <- performance_summary$Method[1:min(3, nrow(performance_summary))]
cat("Selected top", length(top_methods), "models:", paste(top_methods, collapse = ", "), "\n")

# =============================================================================
# SECTION 15: TRAIN FINAL MODELS (FULL DATASET)
# =============================================================================

# Step 11: Train final models using all data (not just training folds)
# These models will be used for ensemble prediction
cat("Training final models for ensemble...\n")

final_models <- list()

for(method in top_methods) {
  cat("Training final", method, "model...\n")
  
  tryCatch({
    # ===========================================
    # RANDOM FOREST (RF)
    # ===========================================
    if(method == "RF") {
      final_models[[method]] <- randomForest::randomForest(
        as.factor(PresAbs) ~ .,
        data = model_data,
        ntree = 1000,
        importance = TRUE
      )
      
      # ===========================================
      # GENERALIZED ADDITIVE MODEL (GAM)
      # ===========================================
    } else if(method == "GAM") {
      formula_gam <- as.formula(paste("PresAbs ~", 
                                      paste("s(", names(model_data)[-1], ", k=3)", collapse = " + ")))
      final_models[[method]] <- mgcv::gam(formula_gam,
                                          data = model_data,
                                          family = binomial,
                                          method = "REML")
      
      # ===========================================
      # GRADIENT BOOSTING MACHINE (GBM)
      # ===========================================
    } else if(method == "GBM") {
      final_models[[method]] <- gbm::gbm(
        PresAbs ~ .,
        data = model_data,
        distribution = "bernoulli",
        n.trees = 1000,
        interaction.depth = 4,
        shrinkage = 0.01
      )
      
      # ===========================================
      # EXTREME GRADIENT BOOSTING (XGBOOST)
      # ===========================================
    } else if(method == "XGBOOST") {
      train_matrix <- xgboost::xgb.DMatrix(
        data = as.matrix(model_data[, -1]), 
        label = model_data$PresAbs
      )
      final_models[[method]] <- xgboost::xgboost(
        data = train_matrix,
        nrounds = 200,
        objective = "binary:logistic",
        eval_metric = "logloss",
        verbose = 0
      )
      
      # ===========================================
      # GENERALIZED LINEAR MODEL (GLM)
      # ===========================================
    } else if(method == "GLM") {
      final_models[[method]] <- glm(PresAbs ~ .,
                                    data = model_data,
                                    family = binomial)
      
      # ===========================================
      # MULTIVARIATE ADAPTIVE REGRESSION SPLINES (MARS)
      # ===========================================
    } else if(method == "MARS") {
      final_models[[method]] <- earth::earth(
        PresAbs ~ .,
        data = model_data,
        glm = list(family = binomial)
      )
      
      # ===========================================
      # SUPPORT VECTOR MACHINE (SVM) - Radial
      # ===========================================
    } else if(method == "SVM_radial") {
      final_models[[method]] <- e1071::svm(
        as.factor(PresAbs) ~ .,
        data = model_data,
        probability = TRUE
      )
      
      # ===========================================
      # FLEXIBLE DISCRIMINANT ANALYSIS (FDA)
      # ===========================================
    } else if(method == "FDA") {
      final_models[[method]] <- mda::fda(
        as.factor(PresAbs) ~ .,
        data = model_data
      )
      
      # ===========================================
      # MAXIMUM ENTROPY (MAXENT)
      # ===========================================
    } else if(method == "MAXENT") {
      final_models[[method]] <- maxnet::maxnet(
        p = model_data$PresAbs,
        data = model_data[, -1]
      )
      
      # ===========================================
      # ARTIFICIAL NEURAL NETWORK (ANN)
      # ===========================================
    } else if(method == "ANN") {
      final_models[[method]] <- nnet::nnet(
        as.factor(PresAbs) ~ .,
        data = model_data,
        size = 7,
        decay = 0.1,
        maxit = 300,
        trace = FALSE
      )
      
      # ===========================================
      # CLASSIFICATION TREE ANALYSIS (CTA)
      # ===========================================
    } else if(method == "CTA") {
      final_models[[method]] <- rpart::rpart(
        as.factor(PresAbs) ~ .,
        data = model_data,
        method = "class",
        control = rpart::rpart.control(cp = 0.01)
      )
      
      # ===========================================
      # SURFACE RANGE ENVELOPE (SRE) - Bioclim
      # ===========================================
    } else if(method == "SRE") {
      final_models[[method]] <- dismo::bioclim(
        x = model_data[model_data$PresAbs == 1, -1]
      )
    }
    
    cat("  -", method, "trained successfully\n")
    
  }, error = function(e) {
    cat("  - Error training", method, ":", e$message, "\n")
  })
}

# =============================================================================
# SECTION 16: CREATE ENSEMBLE PREDICTION
# =============================================================================

# Step 12: Create ensemble prediction from top models
cat("Creating ensemble prediction...\n")

ensemble_predictions <- list()

for(method in names(final_models)) {
  model <- final_models[[method]]
  
  tryCatch({
    # ===========================================
    # RANDOM FOREST PREDICTION
    # ===========================================
    if(method == "RF") {
      pred <- predict(env_stack, model, type = "prob", index = 2)
      
      # ===========================================
      # GAM, GBM, GLM, MARS PREDICTIONS
      # ===========================================
    } else if(method %in% c("GAM", "GBM", "GLM", "MARS")) {
      pred <- predict(env_stack, model, type = "response")
      
      # ===========================================
      # XGBOOST PREDICTION
      # ===========================================
    } else if(method == "XGBOOST") {
      # Convert raster stack to data frame for prediction
      env_df <- as.data.frame(env_stack, xy = TRUE)
      env_df_complete <- env_df[complete.cases(env_df), ]
      
      if(nrow(env_df_complete) > 0) {
        pred_xgb <- predict(model, as.matrix(env_df_complete[, -(1:2)]))
        pred <- rasterFromXYZ(cbind(env_df_complete[, 1:2], pred_xgb))
      } else {
        pred <- NULL
      }
      
      # ===========================================
      # SVM PREDICTION
      # ===========================================
    } else if(method == "SVM_radial") {
      pred <- predict(env_stack, model, probability = TRUE)
      pred <- pred[[2]]  # Extract probability for presence
      
      # ===========================================
      # FDA PREDICTION
      # ===========================================
    } else if(method == "FDA") {
      env_df <- as.data.frame(env_stack, xy = TRUE)
      env_df_complete <- env_df[complete.cases(env_df), ]
      
      if(nrow(env_df_complete) > 0) {
        pred_obj <- predict(model, env_df_complete[, -(1:2)])
        pred_vals <- pred_obj[,2]  # Probability of presence
        pred <- rasterFromXYZ(cbind(env_df_complete[, 1:2], pred_vals))
      } else {
        pred <- NULL
      }
      
      # ===========================================
      # MAXENT PREDICTION
      # ===========================================
    } else if(method == "MAXENT") {
      pred <- predict(env_stack, model, type = "logistic")
      
      # ===========================================
      # ANN PREDICTION
      # ===========================================
    } else if(method == "ANN") {
      pred <- predict(env_stack, model, type = "raw")
      
      # ===========================================
      # CTA PREDICTION
      # ===========================================
    } else if(method == "CTA") {
      pred <- predict(env_stack, model, type = "prob", index = 2)
      
      # ===========================================
      # SRE PREDICTION
      # ===========================================
    } else if(method == "SRE") {
      pred <- predict(env_stack, model)
    }
    
    # Store prediction if successful
    if(!is.null(pred)) {
      ensemble_predictions[[method]] <- pred
      cat("  -", method, "prediction created\n")
    }
    
  }, error = function(e) {
    cat("  - Error creating prediction for", method, ":", e$message, "\n")
  })
}

# =============================================================================
# SECTION 17: CREATE ENSEMBLE MEAN AND SAVE
# =============================================================================

# Step 13: Create simple average ensemble and save
if(length(ensemble_predictions) > 0) {
  cat("Creating ensemble from", length(ensemble_predictions), "models\n")
  ensemble_stack <- stack(ensemble_predictions)
  ensemble_mean <- mean(ensemble_stack, na.rm = TRUE)
  
  # Save ensemble prediction as GeoTIFF
  writeRaster(ensemble_mean, "Ensemble_SDM_Prediction.tif", format = "GTiff", overwrite = TRUE)
  
  # Create ensemble prediction map
  ensemble_df <- as.data.frame(ensemble_mean, xy = TRUE)
  colnames(ensemble_df) <- c("x", "y", "Suitability")
  
  p3 <- ggplot() +
    geom_tile(data = ensemble_df, aes(x = x, y = y, fill = Suitability)) +
    scale_fill_viridis_c(option = "plasma", name = "Habitat\nSuitability") +
    geom_point(data = occ, aes(x = lon, y = lat), color = "red", size = 0.5, alpha = 0.5) +
    labs(title = "Ensemble Habitat Suitability Map",
         subtitle = paste("Top", length(ensemble_predictions), "Models"),
         x = "Longitude", y = "Latitude") +
    theme_minimal() +
    coord_equal()
  
  ggsave("Ensemble_Prediction_Map.png", p3, width = 10, height = 8, dpi = 300)
  
  cat("Ensemble prediction saved successfully!\n")
} else {
  cat("Warning: No predictions could be created for ensemble.\n")
}

# =============================================================================
# SECTION 18: VARIABLE IMPORTANCE ANALYSIS
# =============================================================================

# Step 13: Variable importance analysis (in percentage)
# This section extracts and normalizes variable importance from all final models
# and creates both a heatmap and a boxplot visualization

cat("\n", strrep("=", 60), "\n", sep = "")
cat("VARIABLE IMPORTANCE IN PERCENTAGE\n")
cat(strrep("=", 60), "\n\n", sep = "")

# Get all final models and add Ensemble
all_models <- c(names(final_models), "Ensemble")
available_models <- all_models

cat("Analyzing variable importance for:", paste(available_models, collapse = ", "), "\n\n")

# Create data frame for variable importance
var_imp_data <- data.frame()

# Extract importance for each final model
for(method in names(final_models)) {
  cat("  Extracting importance for", method, "... ")
  
  tryCatch({
    model <- final_models[[method]]
    
    # ===========================================
    # RANDOM FOREST IMPORTANCE
    # ===========================================
    if(method == "RF") {
      imp <- randomForest::importance(model)
      if("MeanDecreaseAccuracy" %in% colnames(imp)) {
        raw_imp <- imp[, "MeanDecreaseAccuracy"]
      } else if("IncNodePurity" %in% colnames(imp)) {
        raw_imp <- imp[, "IncNodePurity"]
      } else {
        raw_imp <- imp[, 1]  # Take first column as fallback
      }
      imp_df <- data.frame(
        Model = method,
        Variable = rownames(imp),
        Raw_Importance = raw_imp,
        stringsAsFactors = FALSE
      )
      
      # ===========================================
      # GBM IMPORTANCE
      # ===========================================
    } else if(method == "GBM") {
      imp <- summary(model, plotit = FALSE)
      imp_df <- data.frame(
        Model = method,
        Variable = imp$var,
        Raw_Importance = imp$rel.inf,
        stringsAsFactors = FALSE
      )
      
      # ===========================================
      # MAXENT IMPORTANCE
      # ===========================================
    } else if(method == "MAXENT") {
      # Try permutation importance first, then betas
      if(!is.null(model$perms)) {
        imp_values <- unname(model$perms)
        var_names <- names(model$perms)
      } else if(!is.null(model$betas)) {
        # Use absolute values of betas
        imp_values <- abs(model$betas)
        var_names <- names(model$betas)
      } else {
        # Fallback: correlation with response
        vars <- names(model_data)[-1]
        imp_values <- sapply(vars, function(var) {
          abs(cor(model_data[[var]], model_data$PresAbs, use = "complete.obs"))
        })
        var_names <- vars
      }
      imp_df <- data.frame(
        Model = method,
        Variable = var_names,
        Raw_Importance = imp_values,
        stringsAsFactors = FALSE
      )
      
      # ===========================================
      # OTHER MODELS (Correlation-based importance)
      # ===========================================
    } else {
      # For other models, calculate correlation-based importance
      vars <- names(model_data)[-1]
      cor_imp <- sapply(vars, function(var) {
        abs(cor(model_data[[var]], model_data$PresAbs, use = "complete.obs"))
      })
      cor_imp[is.na(cor_imp)] <- 0
      imp_df <- data.frame(
        Model = method,
        Variable = vars,
        Raw_Importance = cor_imp,
        stringsAsFactors = FALSE
      )
    }
    
    if(exists("imp_df") && nrow(imp_df) > 0) {
      # Convert raw importance to percentage (sum = 100%)
      imp_df$Percentage <- (imp_df$Raw_Importance / sum(imp_df$Raw_Importance, na.rm = TRUE)) * 100
      var_imp_data <- rbind(var_imp_data, imp_df)
      cat("✓ (", nrow(imp_df), " variables)\n", sep = "")
      rm(imp_df)
    } else {
      cat("✗ (failed)\n")
    }
    
  }, error = function(e) {
    cat("✗ ERROR:", e$message, "\n")
  })
}

# =============================================================================
# SECTION 19: CALCULATE ENSEMBLE IMPORTANCE
# =============================================================================

# Calculate ensemble importance (average across all models)
cat("\n  Calculating ensemble importance... ")
if(nrow(var_imp_data) > 0) {
  # Calculate mean percentage across all models for each variable
  ensemble_imp <- aggregate(Percentage ~ Variable, 
                            data = var_imp_data, 
                            FUN = mean, na.rm = TRUE)
  
  ensemble_df <- data.frame(
    Model = "Ensemble",
    Variable = ensemble_imp$Variable,
    Raw_Importance = NA,
    Percentage = ensemble_imp$Percentage,
    stringsAsFactors = FALSE
  )
  
  var_imp_data <- rbind(var_imp_data, ensemble_df)
  cat("✓\n")
} else {
  cat("✗ (no data)\n")
}

# Save variable importance data to CSV
if(nrow(var_imp_data) > 0) {
  write.csv(var_imp_data, "Variable_Importance_Percentage.csv", row.names = FALSE)
  cat("\n✓ Variable importance saved: Variable_Importance_Percentage.csv\n")
} else {
  cat("\n✗ No variable importance data to save\n")
}

# =============================================================================
# SECTION 20: VARIABLE IMPORTANCE HEATMAP
# =============================================================================

# Create a heatmap showing variable importance across all models
if(nrow(var_imp_data) > 0) {
  cat("\nCreating variable importance heatmap...\n")
  
  # Create heatmap data
  heatmap_data <- var_imp_data[, c("Model", "Variable", "Percentage")]
  
  # Create the heatmap plot
  heatmap_plot <- ggplot(heatmap_data, aes(x = Model, y = Variable, fill = Percentage)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = sprintf("%.1f%%", Percentage)), 
              color = "black", size = 3) +
    scale_fill_viridis_c(option = "plasma", name = "Importance (%)") +
    labs(title = "Variable Importance Heatmap",
         subtitle = "Percentage contribution of each environmental variable",
         x = "Model", y = "Environmental Variable") +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
  
  # Save heatmap plot
  ggsave("Variable_Importance_Heatmap.png", heatmap_plot, width = 10, height = 8, dpi = 300)
  cat("✓ Heatmap saved: Variable_Importance_Heatmap.png\n")
  
  # Display heatmap
  print(heatmap_plot)
}

# =============================================================================
# SECTION 21: VARIABLE IMPORTANCE BOXPLOT
# =============================================================================

# Create boxplot showing distribution of importance across models
library(tidyverse)
cat("\nExtracting and normalizing variable importance from all models...\n")

all_imp_data <- data.frame()

for(method in names(final_models)) {
  cat("  Extracting from", method, "... ")
  
  tryCatch({
    model <- final_models[[method]]
    
    # ===========================================
    # RANDOM FOREST IMPORTANCE
    # ===========================================
    if(method == "RF") {
      imp <- randomForest::importance(model)
      if("MeanDecreaseAccuracy" %in% colnames(imp)) {
        raw_importance <- imp[, "MeanDecreaseAccuracy"]
      } else {
        raw_importance <- imp[, 1]
      }
      var_names <- rownames(imp)
      
      # ===========================================
      # GBM IMPORTANCE
      # ===========================================
    } else if(method == "GBM") {
      imp <- summary(model, plotit = FALSE)
      raw_importance <- imp$rel.inf
      var_names <- imp$var
    } else if(method == "MAXENT") {
      # For MAXENT, try permutation importance first
      if(!is.null(model$perms)) {
        raw_importance <- unname(model$perms)
        var_names <- names(model$perms)
      } else {
        # Fallback: use correlation
        vars <- names(model_data)[-1]
        raw_importance <- sapply(vars, function(var) {
          abs(cor(model_data[[var]], model_data$PresAbs, use = "complete.obs"))
        })
        var_names <- vars
      }
      
      # ===========================================
      # OTHER MODELS (Correlation-based)
      # ===========================================
    } else {
      # For all other models, use absolute correlation
      vars <- names(model_data)[-1]
      raw_importance <- sapply(vars, function(var) {
        abs(cor(model_data[[var]], model_data$PresAbs, use = "complete.obs"))
      })
      var_names <- vars
    }
    
    # Convert to percentage (0-100) for this model
    if(sum(raw_importance, na.rm = TRUE) > 0) {
      importance_pct <- (raw_importance / sum(raw_importance, na.rm = TRUE)) * 100
    } else {
      importance_pct <- rep(0, length(raw_importance))
    }
    
    # Create data frame with percentages
    method_data <- data.frame(
      Model = method,
      Variable = var_names,
      Raw_Importance = raw_importance,
      Importance_Pct = importance_pct,
      stringsAsFactors = FALSE
    )
    
    all_imp_data <- rbind(all_imp_data, method_data)
    cat("✓ (converted to %)\n")
    
  }, error = function(e) {
    cat("✗\n")
  })
}

# Calculate summary statistics using percentages
if(nrow(all_imp_data) > 0) {
  var_summary <- all_imp_data %>%
    group_by(Variable) %>%
    summarise(
      Mean_Importance_Pct = mean(Importance_Pct, na.rm = TRUE),
      SD_Importance_Pct = sd(Importance_Pct, na.rm = TRUE),
      Min_Pct = min(Importance_Pct, na.rm = TRUE),
      Max_Pct = max(Importance_Pct, na.rm = TRUE),
      N_Models = n(),
      .groups = 'drop'
    ) %>%
    mutate(
      # Re-scale to ensure sum = 100% (just in case)
      Final_Pct = Mean_Importance_Pct / sum(Mean_Importance_Pct) * 100
    ) %>%
    arrange(desc(Final_Pct))
  
  # Verify the sum is 100%
  cat("\nSum of final percentages:", sum(var_summary$Final_Pct), "%\n")
  
  # Print summary
  print(var_summary)
  
  # Save to CSV
  write.csv(var_summary, "Variable_Importance_Percentage_Summary.csv", row.names = FALSE)
  cat("\n✓ Variable importance percentage summary saved: Variable_Importance_Percentage_Summary.csv\n")
  
  # Create the boxplot using percentages
  boxplot_plot <- ggplot(all_imp_data, aes(x = reorder(Variable, Importance_Pct, FUN = mean), y = Importance_Pct)) +
    geom_boxplot(fill = "#76B7B2", alpha = 0.8, outlier.shape = NA, width = 0.6) +
    geom_jitter(width = 0.15, size = 2, alpha = 0.6, color = "#E15759") +
    stat_summary(
      fun = mean, geom = "point", 
      shape = 23, size = 4, fill = "#EDC948", color = "black"
    ) +
    geom_text(
      data = var_summary,
      aes(x = Variable, y = max(all_imp_data$Importance_Pct, na.rm = TRUE) * 1.1,
          label = paste0(round(Final_Pct, 1), "%")),
      hjust = -0.1, size = 3.5, fontface = "bold"
    ) +
    labs(
      title = "Variable Importance Distribution Across All Models",
      subtitle = paste("Based on", length(unique(all_imp_data$Model)), "models | All values normalized to percentage (0-100%)",
                       "\nBoxes show IQR, whiskers show range | Yellow diamonds = mean importance"),
      x = "Environmental Variables",
      y = "Importance (%)",
      caption = paste("Generated", format(Sys.Date(), "%Y-%m-%d"))
    ) +
    coord_flip() +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 9),
      plot.caption = element_text(hjust = 1, size = 8, color = "gray50"),
      axis.title = element_text(size = 11),
      axis.text.y = element_text(size = 10, face = "bold"),
      axis.text.x = element_text(size = 9),
      panel.grid.major.x = element_line(color = "gray90"),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(color = "gray95")
    ) +
    scale_y_continuous(
      breaks = seq(0, 100, by = 20), 
      limits = c(0, max(all_imp_data$Importance_Pct, na.rm = TRUE) * 1.2),
      expand = expansion(mult = c(0.05, 0.25)),
      labels = function(x) paste0(x, "%")
    )
  
  # Save the plot
  ggsave("Variable_Importance_Boxplot_Percentage.png", boxplot_plot, width = 12, height = 8, dpi = 300)
  cat("✓ Boxplot with percentages saved: Variable_Importance_Boxplot_Percentage.png\n")
  
  # Display the plot
  print(boxplot_plot)
  
  # Save the data used for the plot
  write.csv(all_imp_data, "Variable_Importance_Boxplot_Data_Percentage.csv", row.names = FALSE)
  cat("✓ Boxplot percentage data saved: Variable_Importance_Boxplot_Data_Percentage.csv\n")
  
} else {
  cat("✗ No variable importance data extracted\n")
}

# =============================================================================
# SECTION 22: RESPONSE CURVES (FACETED)
# =============================================================================

# Step 14: Create faceted response curves for all variables
# Response curves show how predicted suitability changes along each
# environmental gradient while holding other variables at their median

cat("\nCreating faceted response curves with ensemble model...\n")

# Function to generate response curve data
create_faceted_response_data <- function(env_stack, model_data, final_models) {
  
  all_response_data <- data.frame()
  
  # Get models to plot (all final models + ensemble)
  methods_to_plot <- names(final_models)
  
  # For each environmental variable
  for(var_name in names(env_stack)) {
    cat("  Processing", var_name, "... ")
    
    tryCatch({
      # Get range from model data
      var_values <- model_data[[var_name]]
      var_min <- min(var_values, na.rm = TRUE)
      var_max <- max(var_values, na.rm = TRUE)
      
      # Create sequence (30 points for computational efficiency)
      value_seq <- seq(var_min, var_max, length.out = 30)
      
      # Create template data with median values for all other variables
      template_data <- model_data[1, -1]
      template_data[] <- apply(model_data[, -1], 2, median, na.rm = TRUE)
      
      # Store predictions for each method
      predictions <- data.frame(
        Variable = var_name,
        Value = rep(value_seq, length(methods_to_plot)),
        Method = rep(methods_to_plot, each = length(value_seq)),
        Suitability = NA
      )
      
      # Get predictions for each method
      row_idx <- 1
      for(method in methods_to_plot) {
        model <- final_models[[method]]
        
        for(i in 1:length(value_seq)) {
          temp_data <- template_data
          temp_data[[var_name]] <- value_seq[i]
          
          tryCatch({
            if(method == "RF") {
              pred <- predict(model, temp_data, type = "prob")[1, 2]
            } else if(method == "GBM") {
              pred <- predict(model, temp_data, n.trees = 1000, type = "response")
            } else if(method %in% c("GLM", "MARS", "GAM")) {
              pred <- predict(model, temp_data, type = "response")[1]
            } else if(method == "XGBOOST") {
              temp_matrix <- xgboost::xgb.DMatrix(
                data = as.matrix(temp_data)
              )
              pred <- predict(model, temp_matrix)[1]
            } else if(method == "SVM_radial") {
              pred_obj <- predict(model, temp_data, probability = TRUE)
              pred <- attr(pred_obj, "probabilities")[1, 2]
            } else if(method == "MAXENT") {
              pred <- predict(model, temp_data, type = "logistic")[1]
            } else if(method == "ANN") {
              pred <- predict(model, temp_data, type = "raw")[1]
            } else if(method == "CTA") {
              pred <- predict(model, temp_data, type = "prob")[1, 2]
            } else {
              pred <- predict(model, temp_data)[1]
            }
            
            if(is.numeric(pred) && !is.na(pred)) {
              predictions$Suitability[row_idx] <- pred
            }
          }, error = function(e) {
            # Silently skip prediction errors
          })
          
          row_idx <- row_idx + 1
        }
      }
      
      # Add ensemble predictions (average of all models)
      if(exists("ensemble_mean") && !is.null(ensemble_mean)) {
        ensemble_preds <- numeric(length(value_seq))
        
        for(i in 1:length(value_seq)) {
          temp_data <- template_data
          temp_data[[var_name]] <- value_seq[i]
          
          # Get predictions from all models
          model_preds <- numeric(0)
          for(method in methods_to_plot) {
            model <- final_models[[method]]
            try({
              if(method == "RF") {
                pred <- predict(model, temp_data, type = "prob")[1, 2]
              } else if(method == "GBM") {
                pred <- predict(model, temp_data, n.trees = 1000, type = "response")
              } else if(method %in% c("GLM", "MARS", "GAM")) {
                pred <- predict(model, temp_data, type = "response")[1]
              } else if(method == "XGBOOST") {
                temp_matrix <- xgboost::xgb.DMatrix(
                  data = as.matrix(temp_data)
                )
                pred <- predict(model, temp_matrix)[1]
              } else if(method == "SVM_radial") {
                pred_obj <- predict(model, temp_data, probability = TRUE)
                pred <- attr(pred_obj, "probabilities")[1, 2]
              } else if(method == "MAXENT") {
                pred <- predict(model, temp_data, type = "logistic")[1]
              } else if(method == "ANN") {
                pred <- predict(model, temp_data, type = "raw")[1]
              } else if(method == "CTA") {
                pred <- predict(model, temp_data, type = "prob")[1, 2]
              } else {
                pred <- predict(model, temp_data)[1]
              }
              if(is.numeric(pred)) model_preds <- c(model_preds, pred)
            }, silent = TRUE)
          }
          
          if(length(model_preds) > 0) {
            ensemble_preds[i] <- mean(model_preds, na.rm = TRUE)
          }
        }
        
        # Add ensemble to predictions
        ensemble_data <- data.frame(
          Variable = var_name,
          Value = value_seq,
          Method = "Ensemble",
          Suitability = ensemble_preds
        )
        
        predictions <- rbind(predictions, ensemble_data)
      }
      
      # Remove NA values
      predictions <- predictions[complete.cases(predictions), ]
      
      if(nrow(predictions) > 0) {
        all_response_data <- rbind(all_response_data, predictions)
        cat("✓\n")
      } else {
        cat("✗ (no predictions)\n")
      }
      
    }, error = function(e) {
      cat("✗ ERROR:", e$message, "\n")
    })
  }
  
  return(all_response_data)
}

# Generate response curve data
response_data <- create_faceted_response_data(env_stack, model_data, final_models)

# =============================================================================
# SECTION 23: CREATE FACETED RESPONSE CURVE PLOT
# =============================================================================

# Create faceted plot with all variables and models
if(nrow(response_data) > 0) {
  cat("\nCreating faceted response curve plot...\n")
  
  # Set Ensemble color to black and others to colored lines
  response_data$LineType <- ifelse(response_data$Method == "Ensemble", "solid", "dashed")
  response_data$LineSize <- ifelse(response_data$Method == "Ensemble", 1.8, 1.2)
  
  # Create color palette
  all_methods <- unique(response_data$Method)
  ensemble_color <- "black"
  other_methods <- setdiff(all_methods, "Ensemble")
  
  # Use colorblind-friendly palette (Okabe-Ito)
  if(length(other_methods) > 0) {
    cbb_palette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    other_colors <- cbb_palette[1:length(other_methods)]
    color_palette <- c("Ensemble" = ensemble_color, setNames(other_colors, other_methods))
  } else {
    color_palette <- c("Ensemble" = ensemble_color)
  }
  
  # Create faceted plot
  faceted_plot <- ggplot(response_data, aes(x = Value, y = Suitability, 
                                            color = Method, linetype = Method)) +
    geom_line(size = response_data$LineSize, alpha = 0.9) +
    facet_wrap(~ Variable, scales = "free_x", ncol = 3) +
    scale_color_manual(values = color_palette) +
    scale_linetype_manual(values = c("Ensemble" = "solid", 
                                     setNames(rep("dashed", length(other_methods)), other_methods))) +
    labs(title = "Response Curves: All Variables and Models",
         subtitle = "Black solid line = Ensemble model | Colored dashed lines = Individual models",
         x = "Environmental Variable Value",
         y = "Predicted Suitability",
         caption = paste("Total models:", length(unique(response_data$Method)))) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
      strip.text = element_text(face = "bold", size = 14),
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 12),
      axis.text.x = element_text(size = 12),
      axis.text.y = element_text(size = 12),
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 12),
      legend.key.width = unit(1.5, "cm"),
      legend.key.height = unit(0.8, "cm"),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      plot.caption = element_text(size = 10, hjust = 0.5),
      strip.background = element_rect(fill = "gray90", color = "gray70"),
      panel.grid.major = element_line(color = "gray95"),
      panel.grid.minor = element_blank()
    ) +
    guides(
      color = guide_legend(
        nrow = 2, 
        byrow = TRUE,
        override.aes = list(size = 1.5, alpha = 1)
      ),
      linetype = guide_legend(
        nrow = 2, 
        byrow = TRUE,
        override.aes = list(size = 1.5)
      )
    )
  
  # Save plot with higher resolution
  ggsave("Response_Curves_Faceted.png", faceted_plot, 
         width = 14, height = 10, dpi = 300)
  cat("✓ Faceted response curves saved: Response_Curves_Faceted.png\n")
  
  # Save data to CSV
  write.csv(response_data, "Response_Curves_Data.csv", row.names = FALSE)
  cat("✓ Response curve data saved: Response_Curves_Data.csv\n")
  
  # Display plot
  print(faceted_plot)
  
} else {
  cat("✗ No response curve data generated\n")
}

# =============================================================================
# SECTION 24: SCRIPT COMPLETION MESSAGE
# =============================================================================

cat("\n=== SCRIPT COMPLETED SUCCESSFULLY ===\n")
cat("Files created:\n")
cat("- performance.csv\n")
cat("- TSS_vs_AUC_Biomod2_Style.png\n")
cat("- TSS_AUC_Plot_Data.csv\n")
cat("- Model_Performance_Comparison.png\n")
if(exists("ensemble_mean")) {
  cat("- Ensemble_SDM_Prediction.tif\n")
  cat("- Ensemble_Prediction_Map.png\n")
}
cat("- Variable_Importance_Percentage.csv\n")
cat("- Variable_Importance_Heatmap.png\n")
cat("- Variable_Importance_Boxplot_Percentage.png\n")
cat("- Response_Curves_Faceted.png\n")
cat("- Response_Curves_Data.csv\n")

# =============================================================================
# SECTION 25: FUTURE CLIMATE PROJECTIONS
# =============================================================================

# ----------------------------------------------------------------------------
# FUTURE PREDICTION FOR 2050 SCENARIO
# ----------------------------------------------------------------------------
cat("\n", strrep("=", 70), sep = "")
cat("\nFUTURE PREDICTION FOR SSP1_2050x SCENARIO")
cat("\n", strrep("=", 70), "\n\n", sep = "")

# Step 1: Manually load future environmental layers for 2050
scenario_name <- "SSP1_2050x"

cat("Processing scenario:", scenario_name, "\n")
cat("Loading future environmental layers...\n")

# USER MUST UPDATE THESE PATHS TO THEIR ACTUAL FILE LOCATIONS
cat("  Loading future bio01... ")
bio01_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layer 2050/Aligned/aligned_2050Bioclim_Var_1.tif")
cat("✓\n")

cat("  Loading future bio02... ")
bio02_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layer 2050/Aligned/aligned_2050Bioclim_Var_2.tif")
cat("✓\n")

cat("  Loading future bio03... ")
bio03_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layer 2050/Aligned/aligned_2050Bioclim_Var_3.tif")
cat("✓\n")

cat("  Loading future bio05... ")
bio05_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layer 2050/Aligned/aligned_2050Bioclim_Var_5.tif")
cat("✓\n")

cat("  Loading future bio07... ")
bio07_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layer 2050/Aligned/aligned_2050Bioclim_Var_7.tif")
cat("✓\n")

cat("  Loading future bio12... ")
bio12_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layer 2050/Aligned/aligned_2050Bioclim_Var_12.tif")
cat("✓\n")

# Use SAME static layers (water_dist, HF, elv) as current
cat("  Using same water_dist layer... ")
water_dist_future <- water_dist
cat("✓\n")

cat("  Using same HF layer... ")
HF_future <- HF
cat("✓\n")

cat("  Using same elv layer... ")
elv_future <- elv
cat("✓\n")

# Create future raster stack with the 9 layers
future_stack <- stack(bio01_future, bio02_future, bio03_future, 
                      bio05_future, bio07_future, bio12_future, 
                      water_dist_future, HF_future, elv_future)

names(future_stack) <- c("bio01", "bio02", "bio03", "bio05", "bio07", "bio12", 
                         "water_dist", "HF", "elv")

cat("Your 9 future environmental layers:\n")
print(names(future_stack))

# Plot future stack for visual inspection
plot(future_stack)

# Generate ensemble prediction for future scenario
cat("\nGenerating ensemble prediction for future scenario...\n")
scenario_predictions <- list()
successful_models <- 0

for(method in names(final_models)) {
  cat("  Predicting with", method, "... ")
  
  model <- final_models[[method]]
  
  if(method == "RF") {
    pred <- predict(future_stack, model, type = "prob", index = 2)
  } else if(method == "GBM") {
    pred <- predict(future_stack, model, type = "response", n.trees = 1000)
  } else if(method %in% c("GLM", "MARS", "GAM")) {
    pred <- predict(future_stack, model, type = "response")
  } else if(method == "XGBOOST") {
    env_df <- as.data.frame(future_stack, xy = TRUE)
    env_df_complete <- env_df[complete.cases(env_df), ]
    if(nrow(env_df_complete) > 0) {
      pred_xgb <- predict(model, as.matrix(env_df_complete[, -(1:2)]))
      pred <- rasterFromXYZ(cbind(env_df_complete[, 1:2], pred_xgb))
    } else {
      pred <- NULL
    }
  } else if(method == "SVM_radial") {
    pred <- predict(future_stack, model, probability = TRUE)
    pred <- pred[[2]]
  } else if(method == "MAXENT") {
    pred <- predict(future_stack, model, type = "logistic")
  } else {
    pred <- predict(future_stack, model)
  }
  
  if(!is.null(pred)) {
    scenario_predictions[[method]] <- pred
    successful_models <- successful_models + 1
    cat("✓\n")
  } else {
    cat("✗\n")
  }
}

# Create ensemble mean for 2050
cat("\nCreating ensemble mean...\n")
cat("Successful models:", successful_models, "/", length(final_models), "\n")

if(length(scenario_predictions) > 0) {
  ensemble_stack <- stack(scenario_predictions)
  future_ensemble <- mean(ensemble_stack, na.rm = TRUE)
  
  # Save the prediction
  output_file <- paste0("Ensemble_Future_", scenario_name, ".tif")
  writeRaster(future_ensemble, output_file, format = "GTiff", overwrite = TRUE)
  cat("✓ Prediction saved:", output_file, "\n")
  
} else {
  cat("✗ No valid predictions generated\n")
}

cat("\n" , strrep("-", 70), "\n", sep = "")
cat("SCENARIO", scenario_name, "COMPLETE\n")
cat(strrep("-", 70), "\n", sep = "")

# ----------------------------------------------------------------------------
# FUTURE PREDICTION FOR 2070 SCENARIO
# ----------------------------------------------------------------------------
cat("\n", strrep("=", 70), sep = "")
cat("\nFUTURE PREDICTION FOR SSP1_2070x SCENARIO")
cat("\n", strrep("=", 70), "\n\n", sep = "")

# Step 1: Manually load future environmental layers for 2070
scenario_name <- "SSP1_2070x"

cat("Processing scenario:", scenario_name, "\n")
cat("Loading future environmental layers...\n")

# USER MUST UPDATE THESE PATHS TO THEIR ACTUAL FILE LOCATIONS
cat("  Loading future bio01... ")
bio01_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2070/Aligned/aligned_2070bio1.tif")
cat("✓\n")

cat("  Loading future bio02... ")
bio02_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2070/Aligned/aligned_2070bio2.tif")
cat("✓\n")

cat("  Loading future bio03... ")
bio03_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2070/Aligned/aligned_2070bio3.tif")
cat("✓\n")

cat("  Loading future bio05... ")
bio05_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2070/Aligned/aligned_2070bio5.tif")
cat("✓\n")

cat("  Loading future bio07... ")
bio07_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2070/Aligned/aligned_2070bio7.tif")
cat("✓\n")

cat("  Loading future bio12... ")
bio12_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2070/Aligned/aligned_2070bio12.tif")
cat("✓\n")

# Use SAME static layers
cat("  Using same water_dist layer... ")
water_dist_future <- water_dist
cat("✓\n")

cat("  Using same HF layer... ")
HF_future <- HF
cat("✓\n")

cat("  Using same elv layer... ")
elv_future <- elv
cat("✓\n")

# Create future raster stack
future_stack <- stack(bio01_future, bio02_future, bio03_future, 
                      bio05_future, bio07_future, bio12_future, 
                      water_dist_future, HF_future, elv_future)

names(future_stack) <- c("bio01", "bio02", "bio03", "bio05", "bio07", "bio12", 
                         "water_dist", "HF", "elv")

cat("Your 9 future environmental layers:\n")
print(names(future_stack))

plot(future_stack)

# Generate ensemble prediction for 2070
cat("\nGenerating ensemble prediction for future scenario...\n")
scenario_predictions <- list()
successful_models <- 0

for(method in names(final_models)) {
  cat("  Predicting with", method, "... ")
  
  model <- final_models[[method]]
  
  if(method == "RF") {
    pred <- predict(future_stack, model, type = "prob", index = 2)
  } else if(method == "GBM") {
    pred <- predict(future_stack, model, type = "response", n.trees = 1000)
  } else if(method %in% c("GLM", "MARS", "GAM")) {
    pred <- predict(future_stack, model, type = "response")
  } else if(method == "XGBOOST") {
    env_df <- as.data.frame(future_stack, xy = TRUE)
    env_df_complete <- env_df[complete.cases(env_df), ]
    if(nrow(env_df_complete) > 0) {
      pred_xgb <- predict(model, as.matrix(env_df_complete[, -(1:2)]))
      pred <- rasterFromXYZ(cbind(env_df_complete[, 1:2], pred_xgb))
    } else {
      pred <- NULL
    }
  } else if(method == "SVM_radial") {
    pred <- predict(future_stack, model, probability = TRUE)
    pred <- pred[[2]]
  } else if(method == "MAXENT") {
    pred <- predict(future_stack, model, type = "logistic")
  } else {
    pred <- predict(future_stack, model)
  }
  
  if(!is.null(pred)) {
    scenario_predictions[[method]] <- pred
    successful_models <- successful_models + 1
    cat("✓\n")
  } else {
    cat("✗\n")
  }
}

# Create ensemble mean for 2070
cat("\nCreating ensemble mean...\n")
cat("Successful models:", successful_models, "/", length(final_models), "\n")

if(length(scenario_predictions) > 0) {
  ensemble_stack <- stack(scenario_predictions)
  future_ensemble <- mean(ensemble_stack, na.rm = TRUE)
  
  output_file <- paste0("Ensemble_Future_", scenario_name, ".tif")
  writeRaster(future_ensemble, output_file, format = "GTiff", overwrite = TRUE)
  cat("✓ Prediction saved:", output_file, "\n")
  
} else {
  cat("✗ No valid predictions generated\n")
}

cat("\n" , strrep("-", 70), "\n", sep = "")
cat("SCENARIO", scenario_name, "COMPLETE\n")
cat(strrep("-", 70), "\n", sep = "")

# ----------------------------------------------------------------------------
# FUTURE PREDICTION FOR 2090 SCENARIO
# ----------------------------------------------------------------------------
cat("\n", strrep("=", 70), sep = "")
cat("\nFUTURE PREDICTION FOR SSP1_2090x SCENARIO")
cat("\n", strrep("=", 70), "\n\n", sep = "")

# Step 1: Manually load future environmental layers for 2090
scenario_name <- "SSP1_2090x"

cat("Processing scenario:", scenario_name, "\n")
cat("Loading future environmental layers...\n")

# USER MUST UPDATE THESE PATHS TO THEIR ACTUAL FILE LOCATIONS
cat("  Loading future bio01... ")
bio01_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2090/Aligned/aligned_2090bio1.tif")
cat("✓\n")

cat("  Loading future bio02... ")
bio02_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2090/Aligned/aligned_2090bio2.tif")
cat("✓\n")

cat("  Loading future bio03... ")
bio03_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2090/Aligned/aligned_2090bio3.tif")
cat("✓\n")

cat("  Loading future bio05... ")
bio05_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2090/Aligned/aligned_2090bio5.tif")
cat("✓\n")

cat("  Loading future bio07... ")
bio07_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2090/Aligned/aligned_2090bio7.tif")
cat("✓\n")

cat("  Loading future bio12... ")
bio12_future <- raster("D:/Bustard Project/TRY2 with 12 models/Layers 2090/Aligned/aligned_2090bio12.tif")
cat("✓\n")

# Use SAME static layers
cat("  Using same water_dist layer... ")
water_dist_future <- water_dist
cat("✓\n")

cat("  Using same HF layer... ")
HF_future <- HF
cat("✓\n")

cat("  Using same elv layer... ")
elv_future <- elv
cat("✓\n")

# Create future raster stack
future_stack <- stack(bio01_future, bio02_future, bio03_future, 
                      bio05_future, bio07_future, bio12_future, 
                      water_dist_future, HF_future, elv_future)

names(future_stack) <- c("bio01", "bio02", "bio03", "bio05", "bio07", "bio12", 
                         "water_dist", "HF", "elv")

cat("Your 9 future environmental layers:\n")
print(names(future_stack))

plot(future_stack)

# Generate ensemble prediction for 2090
cat("\nGenerating ensemble prediction for future scenario...\n")
scenario_predictions <- list()
successful_models <- 0

for(method in names(final_models)) {
  cat("  Predicting with", method, "... ")
  
  model <- final_models[[method]]
  
  if(method == "RF") {
    pred <- predict(future_stack, model, type = "prob", index = 2)
  } else if(method == "GBM") {
    pred <- predict(future_stack, model, type = "response", n.trees = 1000)
  } else if(method %in% c("GLM", "MARS", "GAM")) {
    pred <- predict(future_stack, model, type = "response")
  } else if(method == "XGBOOST") {
    env_df <- as.data.frame(future_stack, xy = TRUE)
    env_df_complete <- env_df[complete.cases(env_df), ]
    if(nrow(env_df_complete) > 0) {
      pred_xgb <- predict(model, as.matrix(env_df_complete[, -(1:2)]))
      pred <- rasterFromXYZ(cbind(env_df_complete[, 1:2], pred_xgb))
    } else {
      pred <- NULL
    }
  } else if(method == "SVM_radial") {
    pred <- predict(future_stack, model, probability = TRUE)
    pred <- pred[[2]]
  } else if(method == "MAXENT") {
    pred <- predict(future_stack, model, type = "logistic")
  } else {
    pred <- predict(future_stack, model)
  }
  
  if(!is.null(pred)) {
    scenario_predictions[[method]] <- pred
    successful_models <- successful_models + 1
    cat("✓\n")
  } else {
    cat("✗\n")
  }
}

# Create ensemble mean for 2090
cat("\nCreating ensemble mean...\n")
cat("Successful models:", successful_models, "/", length(final_models), "\n")

if(length(scenario_predictions) > 0) {
  ensemble_stack <- stack(scenario_predictions)
  future_ensemble <- mean(ensemble_stack, na.rm = TRUE)
  
  output_file <- paste0("Ensemble_Future_", scenario_name, ".tif")
  writeRaster(future_ensemble, output_file, format = "GTiff", overwrite = TRUE)
  cat("✓ Prediction saved:", output_file, "\n")
  
} else {
  cat("✗ No valid predictions generated\n")
}

cat("\n" , strrep("-", 70), "\n", sep = "")
cat("SCENARIO", scenario_name, "COMPLETE\n")
cat(strrep("-", 70), "\n", sep = "")

# =============================================================================
# SECTION 26: THRESHOLD OPTIMIZATION ANALYSIS
# =============================================================================

cat("\n", strrep("=", 70), sep = "")
cat("\nTHRESHOLD OPTIMIZATION FOR ENSEMBLE HABITAT SUITABILITY")
cat("\n", strrep("=", 70), "\n\n", sep = "")

# Ensure ensemble_mean exists
if(!exists("ensemble_mean")) {
  cat("Loading ensemble prediction...\n")
  if(file.exists("Ensemble_SDM_Prediction.tif")) {
    ensemble_mean <- raster("Ensemble_SDM_Prediction.tif")
  } else {
    stop("Ensemble prediction not found. Please run the ensemble creation first.")
  }
}

# Extract ensemble values at presence points
cat("Extracting ensemble values at presence points...\n")
if(!exists("occ")) {
  stop("Occurrence data not found. Please ensure 'occ' object exists.")
}
presence_vals <- raster::extract(ensemble_mean, occ)
presence_vals <- presence_vals[!is.na(presence_vals)]

# Extract ensemble values at background points
cat("Extracting ensemble values at background points...\n")
if(!exists("bg_points")) {
  cat("Generating background points...\n")
  bg_points <- randomPoints(env_stack, n = length(presence_vals) * 3)
}
bg_vals <- raster::extract(ensemble_mean, bg_points)
bg_vals <- bg_vals[!is.na(bg_vals)]

# Calculate percentiles for thresholds
cat("Calculating percentile thresholds...\n")
percentiles <- seq(5, 95, by = 5)
thresholds <- quantile(presence_vals, probs = percentiles/100, na.rm = TRUE)

# Create the threshold_df data frame
threshold_df <- data.frame(
  Percentile = percentiles,
  Threshold = thresholds,
  Habitat_Area_Pct = NA,
  Commission_Error = NA,
  Omission_Error = NA
)

# Calculate statistics for each threshold
cat("Evaluating each threshold...\n")
for(i in 1:nrow(threshold_df)) {
  threshold <- threshold_df$Threshold[i]
  
  # Calculate habitat area percentage
  all_vals <- values(ensemble_mean)
  all_vals <- all_vals[!is.na(all_vals)]
  habitat_area_pct <- sum(all_vals >= threshold) / length(all_vals) * 100
  
  # Calculate omission error (proportion of presence points below threshold)
  omission_error <- sum(presence_vals < threshold) / length(presence_vals) * 100
  
  # Calculate commission error (proportion of background points above threshold)
  commission_error <- sum(bg_vals >= threshold) / length(bg_vals) * 100
  
  threshold_df$Habitat_Area_Pct[i] <- habitat_area_pct
  threshold_df$Omission_Error[i] <- omission_error
  threshold_df$Commission_Error[i] <- commission_error
}

# Calculate TSS for each threshold
threshold_df$TSS <- (100 - threshold_df$Omission_Error - threshold_df$Commission_Error) / 100

# Find optimal threshold (maximizing TSS)
optimal_idx <- which.max(threshold_df$TSS)
optimal_threshold <- threshold_df$Threshold[optimal_idx]
optimal_percentile <- threshold_df$Percentile[optimal_idx]

# =============================================================================
# SECTION 27: THRESHOLD DECISION ANALYSIS
# =============================================================================

cat("\n" , strrep("=", 60), "\n", sep = "")
cat("THRESHOLD DECISION ANALYSIS\n")
cat(strrep("=", 60), "\n\n", sep = "")

# Display key thresholds
cat("KEY THRESHOLD OPTIONS:\n")
cat(strrep("-", 60), "\n")

# 1. Max TSS
cat("1. MAX TSS (Statistically Optimal):\n")
cat(sprintf("   Percentile: %d%%\n", optimal_percentile))
cat(sprintf("   Threshold value: %.4f\n", optimal_threshold))
cat(sprintf("   TSS: %.3f (Range: 0-1, >0.5 = good)\n", threshold_df$TSS[optimal_idx]))
cat(sprintf("   Habitat area: %.1f%% of study area\n", threshold_df$Habitat_Area_Pct[optimal_idx]))
cat(sprintf("   Omission error: %.1f%% (presences missed)\n", threshold_df$Omission_Error[optimal_idx]))
cat(sprintf("   Commission error: %.1f%% (background as habitat)\n", threshold_df$Commission_Error[optimal_idx]))
cat("   Use when: Balanced approach for general habitat mapping\n\n")

# 2. Equal omission/commission
equal_idx <- which.min(abs(threshold_df$Omission_Error - threshold_df$Commission_Error))
cat("2. EQUAL ERRORS (Balanced Conservation):\n")
cat(sprintf("   Percentile: %d%%\n", threshold_df$Percentile[equal_idx]))
cat(sprintf("   Threshold value: %.4f\n", threshold_df$Threshold[equal_idx]))
cat(sprintf("   Omission = Commission: %.1f%%\n", threshold_df$Omission_Error[equal_idx]))
cat(sprintf("   TSS: %.3f\n", threshold_df$TSS[equal_idx]))
cat(sprintf("   Habitat area: %.1f%% of study area\n", threshold_df$Habitat_Area_Pct[equal_idx]))
cat("   Use when: Equal weight on missing presences vs false habitat\n\n")

# 3. 10% omission (liberal)
if(any(threshold_df$Omission_Error <= 10)) {
  idx_10 <- max(which(threshold_df$Omission_Error <= 10))
  cat("3. MAX 10% OMISSION (Liberal/Inclusive):\n")
  cat(sprintf("   Percentile: %d%%\n", threshold_df$Percentile[idx_10]))
  cat(sprintf("   Threshold value: %.4f\n", threshold_df$Threshold[idx_10]))
  cat(sprintf("   Omission error: %.1f%%\n", threshold_df$Omission_Error[idx_10]))
  cat(sprintf("   Commission error: %.1f%%\n", threshold_df$Commission_Error[idx_10]))
  cat(sprintf("   TSS: %.3f\n", threshold_df$TSS[idx_10]))
  cat(sprintf("   Habitat area: %.1f%% of study area\n", threshold_df$Habitat_Area_Pct[idx_10]))
  cat("   Use when: Finding all potential habitat, survey planning\n\n")
}

# 4. 10% commission (conservative)
if(any(threshold_df$Commission_Error <= 10)) {
  idx_10c <- max(which(threshold_df$Commission_Error <= 10))
  cat("4. MAX 10% COMMISSION (Conservative):\n")
  cat(sprintf("   Percentile: %d%%\n", threshold_df$Percentile[idx_10c]))
  cat(sprintf("   Threshold value: %.4f\n", threshold_df$Threshold[idx_10c]))
  cat(sprintf("   Commission error: %.1f%%\n", threshold_df$Commission_Error[idx_10c]))
  cat(sprintf("   Omission error: %.1f%%\n", threshold_df$Omission_Error[idx_10c]))
  cat(sprintf("   TSS: %.3f\n", threshold_df$TSS[idx_10c]))
  cat(sprintf("   Habitat area: %.1f%% of study area\n", threshold_df$Habitat_Area_Pct[idx_10c]))
  cat("   Use when: Strict protection, limited resources\n\n")
}

# Save threshold results
write.csv(threshold_df, "Threshold_Analysis_Results.csv", row.names = FALSE)
cat("\n✓ Threshold analysis results saved: Threshold_Analysis_Results.csv\n")

# =============================================================================
# SECTION 28: THRESHOLD DECISION PLOT
# =============================================================================

cat("\nCreating decision plot...\n")

# Prepare data for plotting
plot_data <- data.frame(
  Percentile = rep(threshold_df$Percentile, 3),
  Value = c(threshold_df$Omission_Error, 
            threshold_df$Commission_Error,
            threshold_df$TSS * 100),
  Metric = rep(c("Omission Error", "Commission Error", "TSS"), 
               each = nrow(threshold_df))
)

# Create the decision plot
decision_plot <- ggplot(plot_data, aes(x = Percentile, y = Value, color = Metric)) +
  geom_line(size = 1.2) +
  geom_point(data = subset(plot_data, Percentile == optimal_percentile), 
             aes(x = Percentile, y = Value), 
             size = 4, shape = 18, color = "red") +
  geom_vline(xintercept = optimal_percentile, 
             linetype = "dashed", color = "red", size = 0.8, alpha = 0.7) +
  geom_vline(xintercept = threshold_df$Percentile[equal_idx], 
             linetype = "dashed", color = "blue", size = 0.8, alpha = 0.7) +
  scale_color_manual(values = c("Omission Error" = "#E15759", 
                                "Commission Error" = "#4E79A7", 
                                "TSS" = "#59A14F")) +
  labs(
    title = "Threshold Decision Analysis for Bustard Habitat",
    subtitle = paste("Optimal threshold at", optimal_percentile, 
                     "percentile (Threshold =", round(optimal_threshold, 4), ")"),
    x = "Percentile of Presence Values (%)",
    y = "Error Rate (%) / TSS × 100",
    caption = paste("Red line = Max TSS | Blue line = Equal errors |",
                    "Red diamond = optimal values")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray90"),
    panel.grid.minor = element_line(color = "gray95")
  ) +
  annotate("text", x = optimal_percentile, y = max(plot_data$Value) * 0.95,
           label = paste("Optimal:\n", optimal_percentile, "%"), 
           color = "red", hjust = -0.1, size = 3.5) +
  annotate("text", x = threshold_df$Percentile[equal_idx], 
           y = max(plot_data$Value) * 0.85,
           label = paste("Equal errors:\n", threshold_df$Percentile[equal_idx], "%"), 
           color = "blue", hjust = -0.1, size = 3.5)

# Save the plot
ggsave("Threshold_Decision_Plot.png", decision_plot, width = 10, height = 7, dpi = 300)
decision_plot

# =============================================================================
# SECTION 29: SUMMARY STATISTICS AND RECOMMENDATION
# =============================================================================

cat("\n" , strrep("=", 60), "\n", sep = "")
cat("SUMMARY STATISTICS\n")
cat(strrep("=", 60), "\n\n")

cat("Presence values summary:\n")
cat(sprintf("   N presences: %d\n", length(presence_vals)))
cat(sprintf("   Mean suitability: %.3f\n", mean(presence_vals, na.rm = TRUE)))
cat(sprintf("   Median suitability: %.3f\n", median(presence_vals, na.rm = TRUE)))
cat(sprintf("   Range: %.3f - %.3f\n", min(presence_vals, na.rm = TRUE), 
            max(presence_vals, na.rm = TRUE)))
cat("\n")

cat("Background values summary:\n")
cat(sprintf("   N background: %d\n", length(bg_vals)))
cat(sprintf("   Mean suitability: %.3f\n", mean(bg_vals, na.rm = TRUE)))
cat(sprintf("   Median suitability: %.3f\n", median(bg_vals, na.rm = TRUE)))
cat("\n")

# =============================================================================
# SECTION 30: RECOMMENDATION
# =============================================================================

cat("\n" , strrep("=", 60), "\n", sep = "")
cat("RECOMMENDATION\n")
cat(strrep("=", 60), "\n\n")

cat("For your Great Bustard habitat model:\n\n")

cat("1. FOR GENERAL HABITAT MAPPING:\n")
cat(sprintf("   Use %d%% percentile threshold (%.4f)\n", optimal_percentile, optimal_threshold))
cat("   - This maximizes statistical accuracy (TSS)\n")
cat("   - Balances omission and commission errors\n")
cat("   - Good for publications and general distribution maps\n\n")

cat("2. FOR CONSERVATION PLANNING:\n")
if(any(threshold_df$Commission_Error <= 10)) {
  cat(sprintf("   Use %d%% percentile threshold (%.4f)\n", 
              threshold_df$Percentile[idx_10c], threshold_df$Threshold[idx_10c]))
  cat("   - Conservative: <10% commission error\n")
  cat("   - Minimizes false habitat identification\n")
  cat("   - Good for strict protection areas\n\n")
}

cat("3. FOR SURVEY PLANNING:\n")
if(any(threshold_df$Omission_Error <= 10)) {
  cat(sprintf("   Use %d%% percentile threshold (%.4f)\n", 
              threshold_df$Percentile[idx_10], threshold_df$Threshold[idx_10]))
  cat("   - Liberal: <10% omission error\n")
  cat("   - Captures most known presences\n")
  cat("   - Good for finding new populations\n")
}

cat("\nFINAL DECISION GUIDE:\n")
cat("   - If unsure, use the Max TSS threshold (", optimal_percentile, "%)\n")
cat("   - If prioritizing accuracy over coverage, use a higher percentile\n")
cat("   - If prioritizing coverage over accuracy, use a lower percentile\n")

cat("\n" , strrep("-", 60), "\n", sep = "")
cat("✓ Threshold analysis complete\n")
cat("✓ Decision plot saved: Threshold_Decision_Plot.png\n")
cat("✓ Results saved: Threshold_Analysis_Results.csv\n")
cat(strrep("-", 60), "\n\n", sep = "")

# =============================================================================
# SECTION 31: ADDITIONAL DIAGNOSTIC CHECKS
# =============================================================================

# Check distribution of presence values
cat("Checking presence values:\n")
print(summary(presence_vals))
cat("\n5th percentile:", quantile(presence_vals, 0.05))
cat("\n10th percentile:", quantile(presence_vals, 0.10))

# Plot histogram of presence values
hist(presence_vals, main = "Distribution of Presence Suitability Scores",
     xlab = "Suitability", breaks = 30)
abline(v = c(0.495, 0.610), col = c("red", "blue"), lwd = 2)
legend("topright", legend = c("5% (0.495)", "10% (0.610)"), 
       col = c("red", "blue"), lwd = 2)

# =============================================================================
# SECTION 32: SAVE COMPLETE WORKSPACE
# =============================================================================

# Save complete R workspace for reproducibility
cat("\nSaving complete R workspace...\n")
dir.create("Ensemble_saved_R", showWarnings = FALSE)
save.image(file = "Ensemble_saved_R/Complete_Workspace.RData")
cat("✓ All workspace objects saved to: Ensemble_saved_R/Complete_Workspace.RData\n")

# =============================================================================
# SECTION 33: CURRENT VS FUTURE LAYER COMPARISON
# =============================================================================

# This section compares current and future environmental layers to assess
# climate change impacts on the study area

cat("\n", strrep("=", 70), sep = "")
cat("\nCURRENT vs FUTURE LAYER COMPARISON")
cat("\n", strrep("=", 70), "\n\n", sep = "")

# Check if stacks have same layers
if(!all(names(env_stack) == names(future_stack))) {
  cat("WARNING: Layer names don't match!\n")
  cat("Current layers:", names(env_stack), "\n")
  cat("Future layers:", names(future_stack), "\n")
  stop("Layer names must match for comparison")
}

# Create comparison results data frame
comparison_results <- data.frame(
  Layer = names(env_stack),
  Current_Min = NA,
  Current_Max = NA,
  Current_Mean = NA,
  Current_SD = NA,
  Future_Min = NA,
  Future_Max = NA,
  Future_Mean = NA,
  Future_SD = NA,
  Absolute_Change_Min = NA,
  Absolute_Change_Max = NA,
  Mean_Change = NA,
  Percent_Change_Mean = NA,
  Climate_Overlap_Pct = NA,
  stringsAsFactors = FALSE
)

# Compare each layer
cat("Comparing each environmental layer:\n")
cat(strrep("-", 60), "\n")

for(i in 1:nlayers(env_stack)) {
  layer_name <- names(env_stack)[i]
  cat("Processing", layer_name, "...\n")
  
  # Extract values
  current_vals <- values(env_stack[[i]])
  future_vals <- values(future_stack[[i]])
  
  # Remove NAs
  current_vals <- current_vals[!is.na(current_vals)]
  future_vals <- future_vals[!is.na(future_vals)]
  
  if(length(current_vals) > 0 & length(future_vals) > 0) {
    # Calculate statistics
    comparison_results[i, "Current_Min"] <- min(current_vals)
    comparison_results[i, "Current_Max"] <- max(current_vals)
    comparison_results[i, "Current_Mean"] <- mean(current_vals)
    comparison_results[i, "Current_SD"] <- sd(current_vals)
    
    comparison_results[i, "Future_Min"] <- min(future_vals)
    comparison_results[i, "Future_Max"] <- max(future_vals)
    comparison_results[i, "Future_Mean"] <- mean(future_vals)
    comparison_results[i, "Future_SD"] <- sd(future_vals)
    
    # Calculate changes
    comparison_results[i, "Absolute_Change_Min"] <- 
      comparison_results[i, "Future_Min"] - comparison_results[i, "Current_Min"]
    comparison_results[i, "Absolute_Change_Max"] <- 
      comparison_results[i, "Future_Max"] - comparison_results[i, "Current_Max"]
    comparison_results[i, "Mean_Change"] <- 
      comparison_results[i, "Future_Mean"] - comparison_results[i, "Current_Mean"]
    
    # Percent change (avoid division by zero)
    if(comparison_results[i, "Current_Mean"] != 0) {
      comparison_results[i, "Percent_Change_Mean"] <- 
        (comparison_results[i, "Mean_Change"] / abs(comparison_results[i, "Current_Mean"])) * 100
    } else {
      comparison_results[i, "Percent_Change_Mean"] <- NA
    }
    
    # Calculate climate overlap
    overlap_count <- sum(
      future_vals >= comparison_results[i, "Current_Min"] & 
        future_vals <= comparison_results[i, "Current_Max"]
    )
    comparison_results[i, "Climate_Overlap_Pct"] <- 
      overlap_count / length(future_vals) * 100
  }
}

# Print results
cat("\n" , strrep("=", 60), "\n", sep = "")
cat("LAYER COMPARISON RESULTS\n")
cat(strrep("=", 60), "\n\n", sep = "")

print(comparison_results)

# Save to CSV
write.csv(comparison_results, "Current_vs_Future_Layer_Comparison.csv", row.names = FALSE)
cat("\n✓ Comparison results saved: Current_vs_Future_Layer_Comparison.csv\n")

# =============================================================================
# SECTION 34: COMPARISON VISUALIZATION PLOTS
# =============================================================================

cat("\nCreating comparison plots...\n")

# Plot 1: Mean Change Bar Plot
p1 <- ggplot(comparison_results, aes(x = reorder(Layer, Mean_Change), y = Mean_Change, fill = Mean_Change)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = sprintf("%+.2f\n(%+.1f%%)", Mean_Change, Percent_Change_Mean)), 
            hjust = ifelse(comparison_results$Mean_Change > 0, -0.1, 1.1), 
            size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       midpoint = 0, name = "Change") +
  coord_flip() +
  labs(
    title = "Mean Change: Current vs Future (2090)",
    subtitle = "Bars show absolute change, text shows percent change",
    x = "Environmental Layer",
    y = "Mean Change (Future - Current)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none"
  )

# Plot 2: Climate Overlap
p2 <- ggplot(comparison_results, aes(x = reorder(Layer, Climate_Overlap_Pct), y = Climate_Overlap_Pct, fill = Climate_Overlap_Pct)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_hline(yintercept = c(50, 80, 95), linetype = "dashed", alpha = 0.3) +
  geom_text(aes(label = sprintf("%.1f%%", Climate_Overlap_Pct)), 
            hjust = -0.1, size = 3) +
  scale_fill_viridis_c(option = "plasma", direction = -1, name = "Overlap %") +
  coord_flip(ylim = c(0, 105)) +
  labs(
    title = "Climate Space Overlap",
    subtitle = "Percentage of future values within current range",
    x = "Environmental Layer",
    y = "Overlap Percentage (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none"
  )

# Plot 3: Range Comparison
library(tidyr)
range_data <- comparison_results %>%
  select(Layer, Current_Min, Current_Max, Future_Min, Future_Max) %>%
  pivot_longer(cols = -Layer, names_to = "Metric", values_to = "Value")

p3 <- ggplot(range_data, aes(x = Layer, y = Value, color = Metric, group = Metric)) +
  geom_line(size = 1, alpha = 0.7) +
  geom_point(size = 3) +
  scale_color_manual(
    values = c("Current_Min" = "#1F77B4", "Current_Max" = "#1F77B4",
               "Future_Min" = "#FF7F0E", "Future_Max" = "#FF7F0E"),
    labels = c("Current_Min" = "Current Min", "Current_Max" = "Current Max",
               "Future_Min" = "Future Min", "Future_Max" = "Future Max")
  ) +
  labs(
    title = "Range Comparison: Current vs Future",
    subtitle = "Minimum and maximum values for each layer",
    x = "Environmental Layer",
    y = "Value",
    color = "Metric"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

# Save plots
ggsave("Layer_Mean_Change.png", p1, width = 10, height = 6, dpi = 300)
ggsave("Climate_Overlap.png", p2, width = 10, height = 6, dpi = 300)
ggsave("Range_Comparison.png", p3, width = 10, height = 6, dpi = 300)

cat("\n✓ Comparison plots saved:\n")
cat("  - Layer_Mean_Change.png\n")
cat("  - Climate_Overlap.png\n")
cat("  - Range_Comparison.png\n")

# =============================================================================
# SECTION 35: NOVELTY ANALYSIS
# =============================================================================

# Identify novel climate conditions (future outside current range)
cat("\n\nPerforming novelty analysis...\n")

novel_climate_stack <- stack()

for(i in 1:nlayers(env_stack)) {
  layer_name <- names(env_stack)[i]
  cat("  Checking novelty for", layer_name, "...\n")
  
  # Create novelty mask: 1 = novel, 0 = within range, NA = NA
  novel_layer <- future_stack[[i]] < comparison_results$Current_Min[i] |
    future_stack[[i]] > comparison_results$Current_Max[i]
  
  # Convert to binary (1 = novel)
  novel_layer <- reclassify(novel_layer, rbind(c(0, 0), c(1, 1)))
  names(novel_layer) <- paste0(layer_name, "_novel")
  
  novel_climate_stack <- addLayer(novel_climate_stack, novel_layer)
}

# Calculate overall novelty (any variable novel)
overall_novel <- sum(novel_climate_stack, na.rm = TRUE)
overall_novel[overall_novel > 0] <- 1  # Binary: any novelty

# Save novelty raster
writeRaster(overall_novel, "Novel_Climate_Areas.tif", format = "GTiff", overwrite = TRUE)

# Calculate statistics
novel_area_pct <- cellStats(overall_novel, mean) * 100
cat("\nNOVEL CLIMATE ANALYSIS:\n")
cat(strrep("-", 40), "\n")
cat(sprintf("Areas with novel conditions: %.2f%% of study area\n", novel_area_pct))

# Which variables contribute most to novelty?
novel_summary <- data.frame(
  Layer = names(env_stack),
  Novel_Area_Pct = cellStats(novel_climate_stack, mean) * 100,
  stringsAsFactors = FALSE
)
novel_summary <- novel_summary[order(-novel_summary$Novel_Area_Pct), ]

cat("\nVariables contributing to novelty:\n")
print(novel_summary)

write.csv(novel_summary, "Novel_Climate_Summary.csv", row.names = FALSE)
cat("\n✓ Novelty analysis saved:\n")
cat("  - Novel_Climate_Areas.tif\n")
cat("  - Novel_Climate_Summary.csv\n")

# =============================================================================
# SECTION 36: FINAL SUMMARY
# =============================================================================

cat("\n" , strrep("=", 70), "\n", sep = "")
cat("COMPARISON COMPLETE\n")
cat(strrep("=", 70), "\n", sep = "")

cat("\nKey Findings:\n")
cat("1. Most changing variable:", 
    comparison_results$Layer[which.max(abs(comparison_results$Mean_Change))], 
    sprintf("(%+.2f)\n", max(abs(comparison_results$Mean_Change))))
cat("2. Least overlap variable:", 
    comparison_results$Layer[which.min(comparison_results$Climate_Overlap_Pct)], 
    sprintf("(%.1f%% overlap)\n", min(comparison_results$Climate_Overlap_Pct)))
cat("3. Novel climate areas:", sprintf("%.1f%% of study area\n", novel_area_pct))
cat("4. Average overlap:", 
    sprintf("%.1f%% across all variables\n", mean(comparison_results$Climate_Overlap_Pct)))

# =============================================================================
# SECTION 37: SIDE-BY-SIDE CURRENT VS FUTURE PLOTS
# =============================================================================

# Plot current and future layers side by side for visual comparison
par(mfrow=c(9,2), mar=c(2,2,1.5,1))
for(i in 1:nlayers(env_stack)){
  # Get common scale limits for both rasters
  combined_vals <- c(values(env_stack[[i]]), values(future_stack[[i]]))
  combined_vals <- combined_vals[!is.na(combined_vals)]
  common_limits <- range(combined_vals)
  
  # Plot current with common scale
  plot(env_stack[[i]], main=paste("Current", names(env_stack)[i]), 
       zlim=common_limits, col=viridis::viridis(100))
  
  # Plot future with SAME scale
  plot(future_stack[[i]], main=paste("Future", names(future_stack)[i]), 
       zlim=common_limits, col=viridis::viridis(100))
}

# =============================================================================
# SECTION 38: SPATIAL AUTOCORRELATION CHECK
# =============================================================================

# Check if spatial blocks truly address autocorrelation
library(spdep)
library(sp)

coords <- coordinates(occ_sp)
neighbors <- dnearneigh(coords, 0, 50000)  # 50km neighborhood
moran_test <- moran.test(fold_ids, nb2listw(neighbors))
moran_test
cat("Moran's I for spatial autocorrelation:", moran_test$estimate[1], "\n")

# =============================================================================
# END OF SCRIPT
# =============================================================================

cat("\n", strrep("=", 70), "\n", sep = "")
cat("SCRIPT COMPLETED SUCCESSFULLY\n")
cat(strrep("=", 70), "\n", sep = "")