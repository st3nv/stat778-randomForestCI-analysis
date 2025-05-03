results <- readRDS('server_run/noise_level/simulation_results_noise.rds')

summary_df <- data.frame(
  Function = character(),
  noise_level = numeric(),
  test_id = integer(),
  bias = numeric(),
  variance = numeric(),
  mse = numeric()
)

for (name in names(results)) {
  res <- results[[name]]
  # split by last "_", eg "func_ploy_0.1" -> c("func_ploy", "0.1")
  split_name <- unlist(strsplit(name, "_"))
  func <- paste(split_name[1:(length(split_name)-1)], collapse = "_")
  noise_val <- as.numeric(split_name[length(split_name)])
  for(i in 1:length(res$bias_finite)) {
    summary_df <- rbind(summary_df, data.frame(
      Function = func,
      noise_level = noise_val,
      test_id = i,
      bias = res$bias_finite[i],
      variance = res$variance[i],
      mse = res$bias_finite[i]^2 + res$variance[i],
      stringsAsFactors = FALSE
    ))
  }
}

write.csv(summary_df, "server_run/noise_level/results_noise_level.csv", row.names = FALSE)

