library(randomForest)

func_cos <- function(X) { 3 * cos(pi * rowSums(X)) }
func_xor <- function(X) {
  5 * as.numeric(
    ((X[, 1] > 0.6) != (X[, 2] > 0.6)) |
      ((X[, 3] > 0.6) != (X[, 4] > 0.6))
  )
}
func_and <- function(X) {
  10 * as.numeric(
    (X[, 1] > 0.3) & (X[, 2] > 0.3) &
      (X[, 3] > 0.3) & (X[, 4] > 0.3)
  )
}
func_poly_deg1 <- function(X) { X[, 1] + 2 * X[, 2] }
func_poly_deg2 <- function(X) { X[, 1]^2 + 2 * X[, 2]^2 + X[, 1] + 2 * X[, 2] }

generate_noise <- function(n, type = "normal", sigma = 1) {
  if (type == "normal") {
    return(rnorm(n, mean = 0, sd = sigma))
  } else if (type == "cauchy") {
    return(rcauchy(n, location = 0, scale = sigma))
  } else if (type == "exponential") {
    return(rexp(n, rate = 1 / sigma))  # mean = sigma
  } else if (type == "t") {
    df <- 3  # you can adjust this degree of freedom as needed
    return(rt(n, df = df) * sigma)
  } else {
    stop("Unknown noise type")
  }
}

pred_dist <- function(n_train = 50,
                      n_test = 100,
                      n_tree = 100,
                      p = 2,
                      n_sim = 100,
                      true_func,
                      sigma = 1,
                      noise_type = "normal") {
  
  var_estimates <- matrix(NA, nrow = n_test, ncol = n_sim)
  pred_results <- matrix(NA, nrow = n_test, ncol = n_sim)
  pred_results1000 <- matrix(NA, nrow = n_test, ncol = n_sim)
  
  X_test = matrix(runif(n_test * p), n_test, p)
  y_test = true_func(X_test) + generate_noise(n_test, noise_type, sigma)
  
  for (sim in 1:n_sim) {
    set.seed(sim)
    X_train = matrix(runif(n_train * p), n_train, p)
    y_train = true_func(X_train) + generate_noise(n_train, noise_type, sigma)
    
    rf = randomForest(X_train, y_train, ntree = n_tree, keep.inbag = TRUE)
    # ij = randomForestInfJack(rf, X_test, calibrate = TRUE)
    rf_B1000 = randomForest(X_train, y_train, ntree = 1000, keep.inbag = TRUE)
    
    predictions = predict(rf, newdata = X_test)
    predictions_1000 = predict(rf_B1000, newdata = X_test)
    
    # pred_vars = ij$var.hat
    # var_estimates[, sim] = pred_vars
    pred_results1000[, sim] = predictions_1000
    pred_results[, sim] = predictions
  }
  
  return(list(
    pred_results1000 = pred_results1000,
    pred_results = pred_results,
    X_test = X_test
  ))
}

function_p_lst <- list(
  func_cos = list(p = 2, func = func_cos),
  func_xor = list(p = 50, func = func_xor),
  func_and = list(p = 500, func = func_and),
  func_poly_deg1 = list(p = 2, func = func_poly_deg1),
  func_poly_deg2 = list(p = 2, func = func_poly_deg2)
)

noise_types <- c("normal","t","cauchy", "exponential")
n_train_values <- c(25, 50, 100, 200)
n_tree_values <- c(50, 100, 200)

results <- list()
shapiro_summary <- data.frame()

for (func_name in names(function_p_lst)) {
  func_info <- function_p_lst[[func_name]]
  
  for (noise in noise_types) {
    for (n_train in n_train_values) {
      for (n_tree in n_tree_values) {
        
        cat("Running:", func_name, "| Noise:", noise,
            "| n_train:", n_train, "| n_tree:", n_tree, "\n")
        
        res <- pred_dist(
          n_train = n_train,
          n_test = 100,
          n_tree = n_tree,
          p = func_info$p,
          n_sim = 100,
          true_func = func_info$func,
          sigma = 1,
          noise_type = noise
        )
        
        # Run normality tests
        shapiro_pvals_1000 <- apply(res$pred_results1000, 1, function(x) shapiro.test(x)$p.value)
        shapiro_pass_rate_1000 <- mean(shapiro_pvals_1000 > 0.05)
        
        shapiro_pvals_finite <- apply(res$pred_results, 1, function(x) shapiro.test(x)$p.value)
        shapiro_pass_rate_finite <- mean(shapiro_pvals_finite > 0.05)
        
        # Save result set
        key <- paste(func_name, noise, "n", n_train, "ntree", n_tree, sep = "_")
        results[[key]] <- res
        
        # Append summary
        shapiro_summary <- rbind(
          shapiro_summary,
          data.frame(
            Function = func_name,
            Noise = noise,
            N_Train = n_train,
            N_Tree = n_tree,
            Test_Points = length(shapiro_pvals_1000),
            Normal_Pct_1000 = shapiro_pass_rate_1000 * 100,
            Normal_Pct_Finite = shapiro_pass_rate_finite * 100,
            Mean_Pval_1000 = mean(shapiro_pvals_1000),
            Mean_Pval_Finite = mean(shapiro_pvals_finite)
          ))
      }
    }
  }
}

save(results, shapiro_summary, file = "normality_grid_results.RData")
write.csv(shapiro_summary, "normality_summary_grid.csv", row.names = FALSE)
