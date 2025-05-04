results <- readRDS('server_run/correlated/simulation_results_rho_effect.rds')

summary_df <- data.frame(
  Function = character(),
  rho = numeric(),
  test_id = integer(),
  bias = numeric(),
  variance = numeric(),
  mse = numeric()
)

for (name in names(results)) {
  res <- results[[name]]
  split_name <- unlist(strsplit(name, "_rho"))
  func <- split_name[1]
  rho_val <- as.numeric(split_name[2])
  for(i in 1:length(res$bias_finite)) {
    summary_df <- rbind(summary_df, data.frame(
      Function = func,
      rho = rho_val,
      test_id = i,
      bias = res$bias_finite[i],
      variance = res$variance[i],
      mse = res$bias_finite[i]^2 + res$variance[i],
      stringsAsFactors = FALSE
    ))
  }
}



write.csv(summary_df, "server_run/correlated/results_corr.csv", row.names = FALSE)

