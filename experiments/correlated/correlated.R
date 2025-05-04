library(parallel)
library(randomForest)
library(randomForestCI)
library(MASS)

simulate_exp_correlated <- function(n_train = 50,
                                    n_test = 50,
                                    n_tree = 100,
                                    p = 2,
                                    n_sim = 100,
                                    true_func = func_cos,
                                    sigma = 1,
                                    rho = 0.5) {
  var_estimates <- matrix(NA, nrow = n_test, ncol = n_sim)
  pred_results <- matrix(NA, nrow = n_test, ncol = n_sim)
  pred_results1000 <- matrix(NA, nrow = n_test, ncol = n_sim)

  X_test <- matrix(runif(n_test * p), n_test, p)
  Sigma_test <- sigma^2 * (rho ^ abs(outer(1:n_test, 1:n_test, "-")))
  e_test <- mvrnorm(1, mu = rep(0, n_test), Sigma = Sigma_test)
  y_test <- true_func(X_test) + e_test

  for (sim in 1:n_sim) {
    set.seed(sim)
    X_train <- matrix(runif(n_train * p), n_train, p)
    Sigma_train <- sigma^2 * (rho ^ abs(outer(1:n_train, 1:n_train, "-")))
    e_train <- mvrnorm(1, mu = rep(0, n_train), Sigma = Sigma_train)
    y_train <- true_func(X_train) + e_train

    rf <- randomForest(X_train, y_train, ntree = n_tree, keep.inbag = TRUE)
    ij <- randomForestInfJack(rf, X_test, calibrate = TRUE)

    rf_B1000 <- randomForest(X_train, y_train, ntree = 1000, keep.inbag = TRUE)

    predictions <- predict(rf, newdata = X_test)
    predictions_1000 <- predict(rf_B1000, newdata = X_test)

    var_estimates[, sim] <- ij$var.hat
    pred_results[, sim] <- predictions
    pred_results1000[, sim] <- predictions_1000
  }

  true_variance_values <- apply(pred_results1000, 1, var)
  true_variance_finite_values <- apply(pred_results, 1, var)
  bias <- apply(var_estimates, 1, mean) - true_variance_values
  bias_finite <- apply(var_estimates, 1, mean) - true_variance_finite_values
  variance <- apply(var_estimates, 1, var)
  mse <- mean((bias)^2 + variance)
  mse_finite <- mean((bias_finite)^2 + variance)

  return(list(
    mean_bias = mean(bias),
    mean_variance = mean(variance),
    mean_mse = mse,
    mean_bias_finite = mean(bias_finite),
    mean_mse_finite = mse_finite,
    true_variance = true_variance_values,
    true_variance_finite = true_variance_finite_values,
    bias = bias,
    variance = variance,
    var_estimates = var_estimates
  ))
}

# Function-list and rho values
function_p_lst <- list(
  func_cos = list(p = 2, func = func_cos),
  func_xor = list(p = 50, func = func_xor),
  func_and = list(p = 500, func = func_and),
  func_poly_deg2 = list(p = 2, func = func_poly_deg2)
)

rho_list <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

# Create combinations of function/rho
task_list <- expand.grid(func_name = names(function_p_lst), rho = rho_list, stringsAsFactors = FALSE)

# Set up PSOCK cluster
n_cores <- detectCores() - 1
cl <- makePSOCKCluster(n_cores)

# Export required objects and packages to workers
clusterExport(cl, varlist = c("simulate_exp_correlated", "function_p_lst", 
                              "func_cos", "func_xor", "func_and", "func_poly_deg2"),
              envir = environment())
clusterEvalQ(cl, {
  library(randomForest)
  library(randomForestCI)
  library(MASS)
})

# Run tasks in parallel
results_list <- clusterApply(cl, 1:nrow(task_list), function(i) {
  row <- task_list[i, ]
  func_name <- row$func_name
  rho <- row$rho
  func_info <- function_p_lst[[func_name]]
  
  key <- paste(func_name, sprintf("rho%.2f", rho), sep = "_")
  
  result <- simulate_exp_correlated(
    n_train = 100,
    n_test = 100,
    n_tree = 200,
    p = func_info$p,
    n_sim = 200,
    true_func = func_info$func,
    sigma = 1,
    rho = rho
  )
  
  return(list(key = key, result = result))
})

stopCluster(cl)

# Rebuild named list from results
results_correlated_rho <- setNames(
  lapply(results_list, function(x) x$result),
  sapply(results_list, function(x) x$key)
)

saveRDS(results_correlated_rho, "simulation_results_rho_effect.rds")
