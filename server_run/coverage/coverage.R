library(parallel)
library(randomForest)
library(randomForestCI)
library(MASS)


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
func_poly_deg2 <- function(X) { X[, 1]^2 + 2 * X[, 2]^2 + X[, 1] + 2 * X[, 2] }

generate_noise <- function(n, type = "normal", sigma = 1) {
  if (type == "normal") {
    return(rnorm(n, mean = 0, sd = sigma))
  } else if (type == "cauchy") {
    return(rcauchy(n, location = 0, scale = sigma))
  } else if (type == "exponential") {
    return(rexp(n, rate = 1 / sigma))
  } else if (type == "t") {
    df <- 3
    return(rt(n, df = df) * sigma)
  } else {
    stop("Unknown noise type")
  }
}

simulate <- function(n_train = 50,
                     n_test = 200,
                     n_tree = 100,
                     p = 2,
                     n_sim = 500,
                     true_func,
                     sigma = 1,
                     noise_type = "normal") {

  coverage_all <- numeric(n_sim)
  X_test <- matrix(runif(n_test * p), n_test, p)
  y_test_true <- true_func(X_test)
  y_test <- y_test_true + generate_noise(n_test, noise_type, sigma)

  for (sim in 1:n_sim) {
    set.seed(sim)
    X_train <- matrix(runif(n_train * p), n_train, p)
    y_train <- true_func(X_train) + generate_noise(n_train, noise_type, sigma)

    rf <- randomForest(X_train, y_train, ntree = n_tree, keep.inbag = TRUE)
    ij <- randomForestInfJack(rf, X_test, calibrate = TRUE)

    predictions <- predict(rf, newdata = X_test)
    mse <- mean((y_test - predictions)^2)
    pred_vars <- ij$var.hat

    lower <- predictions - 1.64 * sqrt(pred_vars)
    upper <- predictions + 1.64 * sqrt(pred_vars)

    coverage <- mean((y_test >= lower) & (y_test <= upper))
    coverage_ground_truth <- mean((y_test_true >= lower) & (y_test_true <= upper))
    coverage_all[sim] <- coverage
  }

  return(list(
    coverage = coverage_all,
    mse = mse,
    coverage_ground_truth = coverage_ground_truth
  ))
}

function_p_lst <- list(
  func_cos_p2 = list(p = 2, func = func_cos),
  func_xor_p50 = list(p = 50, func = func_xor),
  func_and_p500 = list(p = 500, func = func_and),
  func_poly_deg2_p2 = list(p = 2, func = func_poly_deg2)
)

noise_types <- c("normal", "cauchy", "exponential", "t")
n_train_values <- c(25, 50, 100, 150, 200)
n_tree_values <- c(20, 50, 100, 150, 200)

param_grid <- expand.grid(
  func_name = names(function_p_lst),
  noise = noise_types,
  n_train = n_train_values,
  n_tree = n_tree_values,
  stringsAsFactors = FALSE
)

n_cores <- detectCores() - 1
cl <- makePSOCKCluster(n_cores)

clusterExport(cl, varlist = c("simulate", "generate_noise", "function_p_lst",
                              "func_cos", "func_xor", "func_and", "func_poly_deg2"),
              envir = environment())

clusterEvalQ(cl, {
  library(randomForest)
  library(randomForestCI)
  library(MASS)
})

results_list <- clusterApply(cl, 1:nrow(param_grid), function(i) {
  row <- param_grid[i, ]
  func_info <- function_p_lst[[row$func_name]]
  
  res <- simulate(
    n_train = row$n_train,
    n_test = 250,
    n_tree = row$n_tree,
    p = func_info$p,
    n_sim = 200,
    true_func = func_info$func,
    sigma = 1,
    noise_type = row$noise
  )
  
  list(
    Function = row$func_name,
    Noise = row$noise,
    N_Train = row$n_train,
    N_Tree = row$n_tree,
    Coverage_Mean = mean(res$coverage),
    Coverage_Ground_Truth = mean(res$coverage_ground_truth),
    MSE = mean(res$mse)
  )
})

stopCluster(cl)

coverage_summary <- do.call(rbind, lapply(results_list, as.data.frame))
write.csv(coverage_summary, "coverage_summary.csv", row.names = FALSE)
