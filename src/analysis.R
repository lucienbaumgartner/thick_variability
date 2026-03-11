# Load libraries
library(jsonlite)
library(plotly)
library(dplyr)
library(stringr)
library(pscl)
library(glmnet)

rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the CSV file
df <- read.csv("../output/scores/multiscore.csv", header=T)
colnames(df)

# Create additional predictors
df <- df %>%
  rename(
    targetWord = "target_word",
    isNegated = "is_negated",
    wordCount = "sentence_wordcount",
  ) %>% 
  mutate(dependency = as.factor(dependency),
         conceptType = factor(ifelse(targetWord %in% c("bad", "good", "right", "wrong"), "Thin", "Thick")),
         )

df <- filter(df, isNegated == "False")

# Determine token polarity based on DW
df <- mutate(df, tokenPolarity = factor(case_when(
  (DW > 0 & sentencePolarity == 0) ~ "Negative",
  (DW > 0 & sentencePolarity == 1) ~ "Positive",
  (DW < 0 & sentencePolarity == 1) ~ "Negative",
  (DW < 0 & sentencePolarity == 0) ~ "Positive",
  (DW == 0) ~ "Neutral",
  TRUE ~ NA_character_
), levels = c("Negative", "Neutral", "Positive")))

# Function for determining the statistical mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Type-level polarity labels
typePol <- df %>% group_by(targetWord) %>% summarise(typePolarity = factor(Mode(tokenPolarity)))
typePol %>% arrange(typePolarity) # looks like that worked well!
df <- left_join(df, typePol)

# We are only interested in sentences where 10 < WordCount < 50
#df <- df %>% filter(wordCount > 10, wordCount < 50, NEC > 0.001)

hist(df$NEC_INLP)
hist(df$RSS_INLP)
hist(df$SC)
hist(df$DW)

pca.res <- prcomp(df[c("NEC_INLP", "RSS_INLP")])$x
df$semanticDim <- pca.res[,1]

ggplot(df, aes(x = NEC_INLP, y = DW, color = factor(targetPolarity))) +
  geom_point(alpha = 0.2) +
  facet_grid(~conceptType)

# Run models and compare
model1 <- glm(conceptType ~ NEC_INLP * DW + wordCount + dependency, data = df, family = binomial(link = "logit"))
model2 <- glm(conceptType ~ RSS_INLP * DW + wordCount + dependency, data = df, family = binomial(link = "logit"))
model3 <- glm(conceptType ~ NEC_INLP * SC + wordCount + dependency, data = df, family = binomial(link = "logit"))
model4 <- glm(conceptType ~ RSS_INLP * SC + wordCount + dependency, data = df, family = binomial(link = "logit"))
AIC(model1, model2, model3, model4)
BIC(model1, model2, model3, model4)
lapply(list(model1, model2, model3, model4), pR2) %>% do.call(rbind, .)


# GLMNET
# Convert your predictors to a matrix
x <- model.matrix(conceptType ~ NEC_INLP * DW + DW * tokenPolarity + NEC_INLP * RSS_INLP + wordCount + dependency, data = df)[, -1]  # remove intercept
# Response variable (must be numeric 0/1)
y <- as.numeric(df$conceptType == "Thick")  # or adjust depending on your factor levels
cv_model <- cv.glmnet(x, y, family = "binomial", alpha = 1)  # alpha = 1 for LASSO; alpha = 0 for Ridge
confusion.glmnet(cv_model, newx = x, newy = y)
pred_probs <- predict(cv_model, newx = x, s = "lambda.min", type = "response")
pred_data <- cbind(pred_probs, x)
ggplot(pred_data, aes(x=`NEC_INLP`, y = lambda.min, color = as.factor(TokenPolarityPositive))) +
  geom_point() +
  geom_smooth(method = "loess", se = TRUE)

ggplot(pred_data, aes(x=`DW`, y = lambda.min, color = as.factor(TokenPolarityPositive))) +
  geom_point() +
  geom_smooth(method = "loess", se = TRUE) +
  scale_x_continuous(limits = c(0, max(pred_data["DW"])))


colnames(df)

ggplot(df, aes(Residual.Semantic.Shift..INLP., IG.Target.Word.Contribution)) +
  #geom_point() +
  geom_bin2d() +
  facet_wrap(~targetWord, nrow = 3)

ggplot(df, aes(Residual.Semantic.Shift..INLP., Normalized.Sentiment.Contribution)) +
  #geom_point() +
  geom_bin2d() +
  facet_wrap(~target_word, nrow = 3)

ggplot(df, aes(Normalized.Semantic.Contribution..INLP., Normalized.Sentiment.Contribution)) +
  #geom_point() +
  geom_bin2d() +
  facet_wrap(~target_word, nrow = 3)

scaling_factor <- max(df$Normalized.Semantic.Contribution..INLP., na.rm = TRUE) /
  max(df$Normalized.Sentiment.Contribution, na.rm = TRUE)

# Normalize one variable for plotting
df <- df %>%
  mutate(
    IG_scaled = Normalized.Sentiment.Contribution * scaling_factor  # define this!
  )

# Plot
ggplot(df, aes(x = word_count)) +
  geom_smooth(aes(y = Normalized.Semantic.Contribution..INLP.), color = "red") +
  geom_smooth(aes(y = IG_scaled), color = "blue") +
  scale_y_continuous(
    name = "Residual Semantic Shift (INLP)",
    sec.axis = sec_axis(~ . / scaling_factor, name = "IG Target Word Contribution")
  ) +
  facet_wrap(~target_word, nrow = 3)

ggplot(df, aes(x = word_count, color = type)) +
  geom_smooth(aes(y = Normalized.Semantic.Contribution..INLP.)) +
  #geom_smooth(aes(y = Normalized.Sentiment.Contribution), color = "blue") +
  labs(
    y = "NNEC",
    color = "Metric"
  ) +
  scale_x_continuous(limits = c(10,50))

ggplot(df, aes(x = word_count, color = type)) +
  #geom_smooth(aes(y = Normalized.Semantic.Contribution..INLP.)) +
  geom_smooth(aes(y = Normalized.Sentiment.Contribution)) +
  labs(
    y = "SC",
    color = "Metric"
  ) +
  scale_x_continuous(limits = c(10,50))

ggplot(df, aes(x = word_count, color = type)) +
  geom_smooth(aes(y = Residual.Semantic.Shift..INLP.)) +
  #geom_smooth(aes(y = Normalized.Sentiment.Contribution), color = "blue") +
  labs(
    y = "RSS",
    color = "Metric"
  ) +
  scale_x_continuous(limits = c(10,50))

ggplot(df, aes(x = word_count, color = type)) +
  #geom_smooth(aes(y = Normalized.Semantic.Contribution..INLP.)) +
  geom_smooth(aes(y = IG.Target.Word.Contribution)) +
  labs(
    y = "DW",
    color = "Metric"
  ) +
  scale_x_continuous(limits = c(10,50))

ggplot(df %>% filter(word_count > 10 & word_count < 50), aes(x = Normalized.Sentiment.Contribution, color = type)) +
  #geom_smooth(aes(y = Normalized.Semantic.Contribution..INLP.)) +
  geom_smooth(aes(y = Normalized.Semantic.Contribution..INLP., weight = 1/word_count)) +
  labs(
    y = "NNEC",
    color = "Metric"
  ) 

ggplot(df %>% filter(word_count > 10 & word_count < 50 & Normalized.Semantic.Contribution > 0), aes(x = IG.Target.Word.Contribution, color = type, y = Normalized.Semantic.Contribution, weight = 1/word_count)) +
  #geom_smooth(aes(y = Normalized.Semantic.Contribution..INLP.)) +
  geom_smooth() +
  #geom_point() +
  labs(
    y = "NNEC",
    color = "Metric"
  ) +
  scale_x_continuous(limits=c(0,0.5))



response_matrix <- cbind(
  NNEC  = df$Normalized.Semantic.Contribution..INLP.,
  RSS   = df$Residual.Semantic.Shift..INLP.,
  SC    = df$Normalized.Sentiment.Contribution,
  DW    = df$IG.Target.Word.Contribution
)

manova_model <- manova(response_matrix ~ word_count * type, data = df)
summary(manova_model, test = "Pillai")  # Pillai’s trace is robust

mod_nnec <- lm(NNEC ~ word_count * type, data = df)
mod_rss  <- lm(RSS  ~ word_count * type, data = df)
mod_sc   <- lm(SC   ~ word_count * type, data = df)
mod_dw   <- lm(DW   ~ word_count * type, data = df)

library(ggeffects)
pred_nnec <- ggpredict(mod_nnec, terms = c("word_count [all]", "type"))
pred_rss  <- ggpredict(mod_rss,  terms = c("word_count [all]", "type"))
pred_sc   <- ggpredict(mod_sc,   terms = c("word_count [all]", "type"))
pred_dw   <- ggpredict(mod_dw,   terms = c("word_count [all]", "type"))

ggplot(df, aes(x = word_count)) +
  #geom_point() +
  geom_smooth(aes(y=Residual.Semantic.Shift..INLP.), color = "red") +
  geom_smooth(aes(y=IG.Target.Word.Contribution), color = "blue") +
  facet_wrap(~target_word, nrow = 3)

# Create custom hover text
df$hover <- paste("Sentence:", df$sentence, "<br>Target:", df$target_word)

# Define the sets of variables to plot
plot_vars <- list(
  list(
    x = "Residual.Semantic.Shift",
    y = "Normalized.Semantic.Contribution",
    z = "Normalized.Sentiment.Contribution",
    z_title = "Normalized.Sentiment.Contribution",
    title = "Semantic Shift; Norm Semantic Contribution; Norm Sentiment Contribution"
  ),
  list(
    x = "Residual.Semantic.Shift",
    y = "Normalized.Semantic.Contribution",
    z = "IG.Target.Word.Contribution",
    z_title = "IG.Target.Word.Contribution",
    title = "Semantic Shift; Norm Semantic Contribution; IG Sentiment Contribution"
  ),
  list(
    x = "Residual.Semantic.Shift..INLP.",
    y = "Normalized.Semantic.Contribution..INLP.",
    z = "Normalized.Sentiment.Contribution",
    z_title = "Normalized.Sentiment.Contribution",
    title = "Semantic Shift (INLP); Norm Semantic Contribution (INLP); Norm Sentiment Contribution"
  ),
  list(
    x = "Residual.Semantic.Shift..INLP.",
    y = "Normalized.Semantic.Contribution..INLP.",
    z = "IG.Target.Word.Contribution",
    z_title = "IG.Target.Word.Contribution",
    title = "Semantic Shift (INLP); Norm Semantic Contribution (INLP); IG Sentiment Contribution"
  )
)

# Generate plots
plots <- lapply(plot_vars, function(vars) {
  # Check if columns exist in df, else skip plotting with warning
  cols_present <- all(c(vars$x, vars$y, vars$z) %in% colnames(df))
  if (!cols_present) {
    warning(paste("Skipping plot because some columns are missing:", vars$title))
    return(NULL)
  }
  
  plot_ly(
    data = df,
    x = as.formula(paste0("~`", vars$x, "`")),
    y = as.formula(paste0("~`", vars$y, "`")),
    z = as.formula(paste0("~`", vars$z, "`")),
    color = ~PolarityLabel,
    colors = c("red", "gray", "green"),
    text = ~hover,
    hoverinfo = "text",
    type = "scatter3d",
    mode = "markers",
    marker = list(size = 5)
  ) %>%
    layout(
      title = vars$title,
      scene = list(
        xaxis = list(title = vars$x),
        yaxis = list(title = vars$y),
        zaxis = list(title = vars$z_title)
      )
    )
})

# To view a specific plot, e.g. the first:
plots[[1]]

# Or to view all plots in an R notebook or interactive session, just:
plots

