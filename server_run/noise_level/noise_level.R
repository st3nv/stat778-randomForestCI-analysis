library(parallel)
library(devtools)
install_github("swager/randomForestCI")
library(randomForestCI)
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
func_poly_deg2 <- function(X) {
  X[, 1]^2 + 2 * X[, 2]^2 + X[, 1] + 2 * X[, 2]
}

simulate_exp <- function(n_train = 50,
                         n_test = 50,
                         n_tree = 100,
                         p = 2,
                         n_sim = 100,
                         true_func = func_cos,
                         sigma = 1) {
  var_estimates <- matrix(NA, nrow = n_test, ncol = n_sim)
  pred_results <- matrix(NA, nrow = n_test, ncol = n_sim)
  pred_results1000 <- matrix(NA, nrow = n_test, ncol = n_sim)

  X_test <- matrix(runif(n_test * p), n_test, p)
  y_test <- true_func(X_test) + rnorm(n_test, sd = sigma)

  for (sim in 1:n_sim) {
    set.seed(sim)
    X_train <- matrix(runif(n_train * p), n_train, p)
    y_train <- true_func(X_train) + rnorm(n_train, sd = sigma)

    rf <- randomForest(X_train, y_train, ntree = n_tree, keep.inbag = TRUE)
    ij <- randomForestInfJack(rf, X_test, calibrate = TRUE)
    rf_B1000 <- randomForest(X_train, y_train, ntree = 1000, keep.inbag = TRUE)

    predictions <- predict(rf, newdata = X_test)
    predictions_1000 <- predict(rf_B1000, newdata = X_test)

    pred_vars <- ij$var.hat
    var_estimates[, sim] <- pred_vars
    pred_results1000[, sim] <- predictions_1000
    pred_results[, sim] <- predictions
  }

  true_variance_values <- apply(pred_results1000, 1, var)
  true_variance_finite_values <- apply(pred_results, 1, var)
  bias <- apply(var_estimates, 1, mean) - true_variance_values
  bias_finite <- apply(var_estimates, 1, mean) - true_variance_finite_values
  variance <- apply(var_estimates, 1, var)
  mse <- mean((bias)^2 + variance)
  mse_finite <- mean((bias_finite)^2 + variance)

  list(
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
  )
}

function_p_lst <- list(
  func_cos = list(p = 2, func = func_cos),
  func_xor = list(p = 50, func = func_xor),
  func_and = list(p = 500, func = func_and),
  func_poly_deg2 = list(p = 2, func = func_poly_deg2)
)

noise_level_list <- c(0.1, 0.5, 1, 2, 3)

param_grid <- expand.grid(
  func_name = names(function_p_lst),
  noise_level = noise_level_list,
  stringsAsFactors = FALSE
)

cl <- makePSOCKCluster(parallel::detectCores() - 1)
clusterExport(cl, varlist = c("simulate_exp", "function_p_lst",
                              "func_cos", "func_xor", "func_and", "func_poly_deg2"))
clusterEvalQ(cl, {
  library(randomForest)
  library(randomForestCI)
})

results_list <- clusterApply(cl, 1:nrow(param_grid), function(i) {
  row <- param_grid[i, ]
  func_info <- function_p_lst[[row$func_name]]

  result <- simulate_exp(
    n_train = 100,
    n_test = 100,
    n_tree = 200,
    p = func_info$p,
    n_sim = 200,
    true_func = func_info$func,
    sigma = row$noise_level
  )

  key <- paste(row$func_name, row$noise_level, sep = "_")
  list(name = key, result = result)
})

stopCluster(cl)

results <- setNames(
  lapply(results_list, function(x) x$result),
  sapply(results_list, function(x) x$name)
)

saveRDS(results, "simulation_results_noise.rds")
