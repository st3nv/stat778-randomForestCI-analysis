results <- readRDS('server_run/func_complexity_3/simulation_results.rds')

summary_df <- data.frame(
  Function = character(),
  test_id = integer(),
  bias = numeric(),
  variance = numeric(),
  mse = numeric()
)

for (func in names(results)) {
  res <- results[[func]]
  for(i in 1:length(res$bias_finite)) {
    summary_df <- rbind(summary_df, data.frame(
      Function = func,
      test_id = i,
      bias = res$bias_finite[i],
      variance = res$variance[i],
      mse = res$bias_finite[i]^2 + res$variance[i],
      stringsAsFactors = FALSE
    ))
  }
}


write.csv(summary_df, "server_run/func_complexity_3/results_func_comp.csv", row.names = FALSE)

