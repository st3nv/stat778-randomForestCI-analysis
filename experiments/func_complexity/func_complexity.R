library(parallel)
library(devtools)
install_github("swager/randomForestCI")
library(randomForestCI)
library(randomForest)


func_poly_deg1 <- function(X) { X[, 1] + 2 * X[, 2] }
func_poly_deg2 <- function(X) { X[, 1]^2 + 2 * X[, 2]^2 + X[, 1] + 2 * X[, 2] }
func_poly_deg3 <- function(X) {
  X[, 1]^3 - 2 * X[, 1]^2 * X[, 2] + X[, 1]^2 + 2 * X[, 2]^2 + X[, 1] + 2 * X[, 2]
}
func_poly_deg4 <- function(X) {
  X[, 1]^4 - 3 * X[, 1]^2 * X[, 2]^2 + X[, 1]^3 - 2 * X[, 1]^2 * X[, 2] +
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

  X_test <- matrix(runif(n_test * p, min = -2, max = 2), n_test, p)
  y_test <- true_func(X_test) + rnorm(n_test, sd = sigma)

  for (sim in 1:n_sim) {
    set.seed(sim)
    X_train <- matrix(runif(n_train * p, min = -2, max = 2), n_train, p)
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
    bias_finite = bias_finite,
    variance = variance,
    var_estimates = var_estimates
  )
}

function_p_lst <- list(
  func_poly_deg1 = list(p = 2, func = func_poly_deg1),
  func_poly_deg2 = list(p = 2, func = func_poly_deg2),
  func_poly_deg3 = list(p = 2, func = func_poly_deg3),
  func_poly_deg4 = list(p = 2, func = func_poly_deg4)
)

param_list <- lapply(names(function_p_lst), function(name) {
  list(name = name, p = function_p_lst[[name]]$p, func = function_p_lst[[name]]$func)
})

cl <- makePSOCKCluster(parallel::detectCores() - 1)
clusterExport(cl, varlist = c("simulate_exp", "func_poly_deg1", "func_poly_deg2",
                              "func_poly_deg3", "func_poly_deg4", "function_p_lst"))
clusterEvalQ(cl, {
  library(randomForest)
  library(randomForestCI)
})

results_list <- clusterApply(cl, param_list, function(param) {
  result <- simulate_exp(
    n_train = 100,
    n_test = 100,
    n_tree = 200,
    p = param$p,
    n_sim = 500,
    true_func = param$func,
    sigma = 1
  )
  list(name = param$name, result = result)
})

stopCluster(cl)

results <- setNames(
  lapply(results_list, function(x) x$result),
  sapply(results_list, function(x) x$name)
)

saveRDS(results, "simulation_results.rds")
