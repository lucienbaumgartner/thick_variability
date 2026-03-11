library(jsonlite)
library(dplyr)
library(ggplot2)
library(gridExtra)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the JSON file
data <- fromJSON("../output/scores/scores.json")

# Convert to data frame
df <- as.data.frame(data)

# Define the sets of variables to plot (pairs only, ignoring z)
plot_pairs <- list(
  list(x = "Residual Semantic Shift", y = "Normalized Semantic Contribution", title = "Residual Semantic Shift vs\n Normalized Semantic Contribution"),
  list(x = "Residual Semantic Shift", y = "Normalized Sentiment Contribution", title = "Residual Semantic Shift vs\n Normalized Sentiment Contribution"),
  list(x = "Normalized Semantic Contribution", y = "Normalized Sentiment Contribution", title = "Normalized Semantic Contribution vs\n Normalized Sentiment Contribution"),
  
  list(x = "Residual Semantic Shift", y = "IG Target Word Contribution", title = "Residual Semantic Shift vs\n IG Target Word Contribution"),
  list(x = "Normalized Semantic Contribution", y = "IG Target Word Contribution", title = "Normalized Semantic Contribution vs\n IG Target Word Contribution"),
  list(x = "Normalized Sentiment Contribution", y = "IG Target Word Contribution", title = "Normalized Sentiment Contribution vs\n IG Target Word Contribution"),
  
  list(x = "Residual Semantic Shift (INLP)", y = "Normalized Semantic Contribution (INLP)", title = "Residual Semantic Shift (INLP) vs\n Normalized Semantic Contribution (INLP)"),
  list(x = "Residual Semantic Shift (INLP)", y = "Normalized Sentiment Contribution", title = "Residual Semantic Shift (INLP) vs\n Normalized Sentiment Contribution"),
  list(x = "Normalized Semantic Contribution (INLP)", y = "Normalized Sentiment Contribution", title = "Normalized Semantic Contribution (INLP) vs\n Normalized Sentiment Contribution"),
  
  list(x = "Residual Semantic Shift (INLP)", y = "IG Target Word Contribution", title = "Residual Semantic Shift (INLP) vs\n IG Target Word Contribution"),
  list(x = "Normalized Semantic Contribution (INLP)", y = "IG Target Word Contribution", title = "Normalized Semantic Contribution (INLP) vs\n IG Target Word Contribution")
)

# Function to generate a 2D density plot for a given pair
library(rlang)  # for sym()

plot_2d_density <- function(df, xvar, yvar, title) {
  if (!(xvar %in% colnames(df)) | !(yvar %in% colnames(df))) {
    warning(paste("Skipping plot, columns missing:", xvar, yvar))
    return(NULL)
  }
  
  x_sym <- sym(xvar)
  y_sym <- sym(yvar)
  
  p <- ggplot(df, aes(x = !!x_sym, y = !!y_sym)) +
    geom_point(alpha = 0.4, size = 1, color = "blue") +
    stat_density_2d(aes(fill = ..level..), alpha = 0.5) +
    scale_fill_viridis_c() +
    theme_minimal() +
    labs(title = title, x = xvar, y = yvar, fill = "Density") +
    theme(legend.position = "right")
  
  return(p)
}


# Generate plots
plots <- lapply(plot_pairs, function(p) plot_2d_density(df, p$x, p$y, p$title))

# Remove NULL plots (missing columns)
plots <- Filter(Negate(is.null), plots)

combined_plot <- do.call(
  grid.arrange,
  c(plots, ncol = 2)
)

ggsave(
  filename = "../output/figures/metric_comparison.png",
  plot = combined_plot,
  width = 10,
  height = 20,
  dpi = 300
)
