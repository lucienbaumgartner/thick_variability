# Load libraries
library(jsonlite)
library(plotly)
library(dplyr)
library(stringr)
library(pscl)
library(glmnet)
library(lme4)
library(brms)
library(reshape2)
library(tidybayes)
library(ggplot2)
library(dplyr)
library(forcats)  # for ordering factors nicely
library(ggdist) 
library(tidyr)
library(ggtext)
library(xtable)

rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the CSV file
df <- read.csv("../output/scores/abstracted_scores.csv", header=T)
df <- df %>% mutate(isNegated = ifelse(is_negated == "True", TRUE, FALSE))
df <- df %>% select(-is_negated)

# Get metadata
meta_controlled <- fromJSON("../input/proof-of-concept_data_controlled.json") %>% as.data.frame
meta_controlled <- melt(meta_controlled, id.vars = "template_id", value.name = "sentence", variable.name = "isNegated") %>% 
  rename(id = "template_id") %>% 
  mutate(isNegated = ifelse(isNegated == "sentence_neg", TRUE, FALSE))

target_terms <- c("compassionate", "cruel", "generous", "selfish", "sincere", "deceitful", "courageous", "cowardly", "virtuous", "vicious")

meta_controlled <- lapply(target_terms, function(term){
  tibble(id = meta_controlled$id, sentence = gsub("TERM", term, meta_controlled$sentence), target_word = term, isNegated = meta_controlled$isNegated)
})
meta_controlled <- do.call(rbind, meta_controlled)
meta_controlled$pair <- NA

meta_naturalistic <- fromJSON("../input/proof-of-concept_data_naturalistic_w_symmetric_neg.json") %>% as.data.frame
meta_naturalistic <- melt(meta_naturalistic, id.vars = c("pair", "frame_id"), value.name = "sentence", variable.name = "isNegated") %>% 
  rename(id = "frame_id") %>% 
  mutate(isNegated = ifelse(isNegated == "sentence_neg", TRUE, FALSE))

meta_naturalistic <- lapply(1:nrow(meta_naturalistic), function(r_index){
  .row <- meta_naturalistic[r_index,]
  terms <- unlist(strsplit(.row$pair, "/"))
  .row1 <- .row %>% mutate(sentence = gsub("TERM", terms[1], sentence), target_word = terms[1])
  .row2 <- .row %>% mutate(sentence = gsub("TERM", terms[2], sentence), target_word = terms[2])
  return(rbind(.row1, .row2))
}) %>% do.call(rbind, .)

meta <- rbind(meta_controlled %>% mutate(corpus = "controlled"), meta_naturalistic %>% mutate(corpus = "naturalistic"))

check <- anti_join(meta, df)
# one wrongly labeled negation
df <- left_join(df %>% select(-isNegated), meta)

colnames(df)
#df <- df %>% filter(target_word %in% c("honest", "dishonest"))
#df$item_id = as.factor(rep(rep(seq_along(1:(nrow(df)/4)), each=2), 2))

# Create additional predictors
df <- df %>%
  rename(
    targetWord = "target_word",
    wordCount = "sentence_wordcount",
  ) %>% 
  mutate(across(c(targetWord, dependency, isNegated), ~ as.factor(.)))

# Determine token polarity based on DW
df <- mutate(df, 
  conceptType = as.factor(case_when(
    targetWord %in% c("cruel", "selfish", "deceitful", "cowardly", "vicious") ~ "NTT",
    targetWord %in% c("compassionate", "generous", "sincere", "courageous", "virtuous") ~ "PTT",
    TRUE ~ NA_character_
  )))

any(is.na(df$conceptType))
df %>% filter(is.na(conceptType))

all(table(df$isNegated)==200)

dfl <- split(df, df$corpus)
dfl <- lapply(dfl, function(x) x %>% mutate(id = as.factor(id)))
str(dfl)

ggplot(dfl[["controlled"]], aes(x = NEC_INLP, fill = isNegated)) +
  geom_density(alpha = 0.4) +
  facet_grid(~conceptType)

ggplot(dfl[["naturalistic"]], aes(x = NEC_INLP, fill = isNegated)) +
  geom_density(alpha = 0.4) +
  facet_grid(~conceptType)

ggplot(dfl[["controlled"]], aes(x = RSS_INLP, fill = isNegated)) +
  geom_density(alpha = 0.4) +
  facet_grid(~conceptType)

ggplot(dfl[["naturalistic"]], aes(x = RSS_INLP, fill = isNegated)) +
  geom_density(alpha = 0.4) +
  facet_grid(~conceptType)

# Data frame `df` has: NEC, SC, Polarity, Sentence, Word

# Multivariate formula

#### Fully controlled corpus

dfl[["controlled"]] <- dfl[["controlled"]] %>%
  mutate(
    NEC_INLP_z = as.numeric(scale(NEC_INLP)),
    RSS_INLP_z = as.numeric(scale(RSS_INLP)),
    SC_signed_z = as.numeric(scale(SC_signed)),
    cw_SC_signed_z = as.numeric(scale(cw_SC_signed))
  )

bf_nec <- bf(
  NEC_INLP_z ~ conceptType * isNegated +
    (1 + isNegated | id) +
    (1 | targetWord) +
    (1 | dependency),
  family = gaussian()
)

bf_rss <- bf(
  RSS_INLP_z ~ conceptType * isNegated +
    (1 + isNegated | id) +
    (1 | targetWord) +
    (1 | dependency),
  family = gaussian()
)

bf_lc <- bf(
  SC_signed_z ~ conceptType * isNegated +
    (1 + isNegated | id) +
    (1 | targetWord) +
    (1 | dependency),
  family = gaussian()
)

bf_cwsc <- bf(
  cw_SC_signed_z ~ conceptType * isNegated +
    (1 + isNegated | id) +
    (1 | targetWord) +
    (1 | dependency),
  family = gaussian()
)

df_c <- dfl[["controlled"]]
fit_c <- brm(
  bf_nec + bf_rss + bf_lc + bf_cwsc,
  data = df_c,
  chains = 4, 
  cores = 4, 
  iter = 4000,
  warmup = 2000,
  seed = 3879,
  file = "../output/models/c1_all_metrics.rds",
  control = list(adapt_delta = 0.99, max_treedepth = 15)
)

summary(fit_c)

# Posterior draws of NEC and SC for each combination of conceptType and isNegated
# Define all combinations of conceptType and isNegated
newdata <- expand.grid(
  conceptType = c("PTT", "NTT"),
  isNegated   = c("FALSE", "TRUE")
)

# Get posterior draws
draws_c <- fit_c %>%
  epred_draws(newdata = newdata, re_formula = NA)  # Use re_formula = NA to ignore any random effects (none here)
draws_c <- draws_c %>%
  rename(Response = .category)

draws_c <- draws_c %>% 
  ungroup() %>% 
  mutate(
    isNegated = ifelse(isNegated == "TRUE", "Under negation", "Affirmation"),
    corpus = "controlled",
    Response = recode(
      Response,
      "cwSCsignedz" = "CWC",
      "SCsignedz"  = "LC",
      "NECINLPz" = "NEC",
      "RSSINLPz" = "RSS"
    )
  )

# 1. condition-level posterior means
cond_means_c <- draws_c %>%
  group_by(.draw, Response, conceptType, isNegated) %>%
  summarise(mu = mean(.epred), .groups = "drop")

nec_table_c <- cond_means_c %>%
  filter(Response == "NEC") %>%
  group_by(.draw, isNegated, conceptType) %>%
  summarise(mu = mean(mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptType, values_from = mu) %>%
  mutate(diff = NTT - PTT) %>%
  group_by(Response = "NEC", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

rss_table_c <- cond_means_c %>%
  filter(Response == "RSS") %>%
  group_by(.draw, isNegated, conceptType) %>%
  summarise(mu = mean(mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptType, values_from = mu) %>%
  mutate(diff = NTT - PTT) %>%
  group_by(Response = "RSS", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

# 2. LC: absolute magnitude (distance from zero), then compare types
lc_table_c <- cond_means_c %>%
  filter(Response == "LC") %>%
  mutate(abs_mu = abs(mu)) %>%
  group_by(.draw, isNegated, conceptType) %>%
  summarise(abs_mu = mean(abs_mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptType, values_from = abs_mu) %>%
  mutate(diff = NTT - PTT) %>%
  group_by(Response = "LC", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

# 3. CWC: signed contrasts
cwc_table_c <- cond_means_c %>%
  filter(Response == "CWC") %>%
  group_by(.draw, isNegated, conceptType) %>%
  summarise(mu = mean(mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptType, values_from = mu) %>%
  mutate(diff = NTT - PTT) %>%
  group_by(Response = "CWC", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

# 4. combine
summary_table_c <- bind_rows(nec_table_c, rss_table_c, lc_table_c, cwc_table_c)


#### Naturalistic corpus

dfl[["naturalistic"]] <- dfl[["naturalistic"]] %>%
  mutate(
    NEC_INLP_z = as.numeric(scale(NEC_INLP)),
    RSS_INLP_z = as.numeric(scale(RSS_INLP)),
    SC_signed_z = as.numeric(scale(SC_signed)),
    cw_SC_signed_z = as.numeric(scale(cw_SC_signed))
  )

bf_nec <- bf(
  NEC_INLP_z ~ conceptType * isNegated +
    (1 | pair) +
    (1 + isNegated | pair:id) +
    (1 | pair:targetWord) +
    (1 | dependency),
  family = gaussian()
)

bf_rss <- bf(
  RSS_INLP_z ~ conceptType * isNegated +
    (1 | pair) +
    (1 + isNegated | pair:id) +
    (1 | pair:targetWord) +
    (1 | dependency),
  family = gaussian()
)

bf_sc <- bf(
  SC_signed_z ~ conceptType * isNegated +
    (1 | pair) +
    (1 + isNegated | pair:id) +
    (1 | pair:targetWord) +
    (1 | dependency),
  family = gaussian()
)

bf_cwsc <- bf(
  cw_SC_signed_z ~ conceptType * isNegated +
    (1 | pair) +
    (1 + isNegated | pair:id) +
    (1 | pair:targetWord) +
    (1 | dependency),
  family = gaussian()
)

df_n <- dfl[["naturalistic"]] %>% mutate(pair = as.factor(pair))
fit_n <- brm(
  bf_nec + bf_rss + bf_sc + bf_cwsc,
  data = df_n,
  chains = 4, 
  cores = 4, 
  iter = 4000,
  warmup = 2000,
  seed = 3879,
  file = "../output/models/c2_all_metrics.rds",
  control = list(adapt_delta = 0.99, max_treedepth = 15)
)

summary(fit_n)

# Posterior draws of NEC and SC for each combination of conceptType and isNegated
# Define all combinations of conceptType and isNegated
newdata <- expand.grid(
  conceptType = c("PTT", "NTT"),
  isNegated   = c("FALSE", "TRUE")
)

# Get posterior draws
draws_n <- fit_n %>%
  epred_draws(newdata = newdata, re_formula = NA)  # Use re_formula = NA to ignore any random effects (none here)
draws_n <- draws_n %>%
  rename(Response = .category)

draws_n <- draws_n %>% 
  ungroup() %>% 
  mutate(
    isNegated = ifelse(isNegated == "TRUE", "Under negation", "Affirmation"),
    corpus = "controlled",
    Response = recode(
      Response,
      "cwSCsignedz" = "CWC",
      "SCsignedz"  = "LC",
      "NECINLPz" = "NEC",
      "RSSINLPz" = "RSS"
    )
  )


# 1. condition-level posterior means
cond_means_n <- draws_n %>%
  group_by(.draw, Response, conceptType, isNegated) %>%
  summarise(mu = mean(.epred), .groups = "drop")

nec_table_n <- cond_means_n %>%
  filter(Response == "NEC") %>%
  group_by(.draw, isNegated, conceptType) %>%
  summarise(mu = mean(mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptType, values_from = mu) %>%
  mutate(diff = NTT - PTT) %>%
  group_by(Response = "NEC", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

rss_table_n <- cond_means_n %>%
  filter(Response == "RSS") %>%
  group_by(.draw, isNegated, conceptType) %>%
  summarise(mu = mean(mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptType, values_from = mu) %>%
  mutate(diff = NTT - PTT) %>%
  group_by(Response = "RSS", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )


# 2. LC: absolute magnitude (distance from zero), then compare types
lc_table_n <- cond_means_n %>%
  filter(Response == "LC") %>%
  mutate(abs_mu = abs(mu)) %>%
  group_by(.draw, isNegated, conceptType) %>%
  summarise(abs_mu = mean(abs_mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptType, values_from = abs_mu) %>%
  mutate(diff = NTT - PTT) %>%
  group_by(Response = "LC", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

# 3. CWC: signed contrasts
cwc_table_n <- cond_means_n %>%
  filter(Response == "CWC") %>%
  group_by(.draw, isNegated, conceptType) %>%
  summarise(mu = mean(mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptType, values_from = mu) %>%
  mutate(diff = NTT - PTT) %>%
  group_by(Response = "CWC", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

# 4. combine
summary_table_n <- bind_rows(nec_table_n, rss_table_n, lc_table_n, cwc_table_n)

summary_table <- bind_rows(summary_table_c, summary_table_n)

xtable(summary_table, digits = 4)

plotdata <- rbind(draws_c, draws_n)
plotdata <- plotdata %>% mutate(
  corpus = ifelse(corpus == "controlled", "<i>C<sub>1</sub></i>: Symmetric", "<i>C<sub>2</sub></i>: Inter-pair Variation"),
  conceptType = ifelse(conceptType == "PTT", "Positive Thick", "Negative Thick")
)

p <- ggplot(plotdata, aes(x = .epred, y = Response, fill = conceptType)) +
  stat_halfeye(.width = c(0.66, 0.95), size = 2, alpha = 0.8) +
  scale_fill_manual(
    values = MetBrewer::met.brewer("Lakota", n = length(unique(plotdata$Response)))
  ) +
  theme_light() +
  theme(panel.spacing = unit(1, "lines"),
        plot.caption = element_markdown(hjust = 0, size = 5),
        strip.text.x = element_markdown(size = 10, margin = margin(0.2,0,0,0, "cm")),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 10),
        legend.position = "top"
        ) +
  facet_grid(isNegated~corpus) +
  labs(fill = "Term Type",
       x = "Predicted Response",
       y = NULL,
       caption = paste0(
         "<i>C<sub>1</sub></i>: bf((LC,CWC) ~ conceptType × isNegated + (1 + isNegated | id) + (1 | targetTerm) + (1 | dependency), family = gaussian())<br>",
         "<i>C<sub>2</sub></i>: bf((LC,CWC) ~ conceptType × isNegated + (1 | pair) + (1 + isNegated | pair:id) + (1 | pair:targetTerm) +
        (1 | dependency), family = gaussian())"
       )) +
  scale_y_discrete(expand = c(0,0.1)) +
  scale_x_continuous(limits = c(-1.25,1.25), breaks = c(-1,0,1), expand = c(0,0))
p

ggsave(p, file = "../output/figures/pilot.png", width = 6, height = 4, dpi = 600)


hypothesis(
  fit,
  "SCsigned_conceptTypePositive - NECINLP_conceptTypePositive = 0"
)

post <- as_draws_df(fit)

# compute conditional means manually
mu_neg_nonneg <- post$b_SCsigned_Intercept
mu_pos_nonneg <- post$b_SCsigned_Intercept + post$b_SCsigned_conceptTypePositive

mu_neg_negated <- post$b_SCsigned_Intercept + post$b_SCsigned_isNegatedTrue
mu_pos_negated <- post$b_SCsigned_Intercept +
  post$b_SCsigned_conceptTypePositive +
  post$b_SCsigned_isNegatedTrue +
  post$`b_SCsigned_conceptTypePositive:isNegatedTrue`

d_neg_nonneg <- abs(mu_neg_nonneg)
d_pos_nonneg <- abs(mu_pos_nonneg)

d_neg_negated <- abs(mu_neg_negated)
d_pos_negated <- abs(mu_pos_negated)

# non-negated contrast
mean(d_pos_nonneg - d_neg_nonneg < 0)

# negated contrast
mean(d_pos_negated - d_neg_negated < 0)


post <- as_draws_df(fit)
# non-negated
mu_neg_nonneg <- post$b_cwSCsigned_Intercept
mu_pos_nonneg <- post$b_cwSCsigned_Intercept + post$b_cwSCsigned_conceptTypePositive

# negated
mu_neg_negated <- post$b_cwSCsigned_Intercept + post$b_cwSCsigned_isNegatedTrue
mu_pos_negated <- post$b_cwSCsigned_Intercept +
  post$b_cwSCsigned_conceptTypePositive +
  post$b_cwSCsigned_isNegatedTrue +
  post$`b_cwSCsigned_conceptTypePositive:isNegatedTrue`
diff_nonneg <- mu_pos_nonneg - mu_neg_nonneg
diff_neg <- mu_pos_negated - mu_neg_negated
mean(diff_nonneg < 0)  # positive < negative?
mean(diff_neg < 0)


p <- ggplot(draws, aes(x = .epred, y = Response, fill = conceptTypeL)) +
  stat_halfeye(.width = c(0.66, 0.95), size = 2, alpha = 0.8) +
  scale_fill_manual(
    values = MetBrewer::met.brewer("Lakota", n = length(unique(draws$Response)))
  ) +
  labs(
    x = "Predicted Response",
    y = NULL
  ) +
  theme_light() +
  theme(panel.spacing = unit(1, "lines")) +
  facet_wrap(~isNegated) +
  labs(fill = "Thick Term") +
  scale_y_discrete(expand = c(0,0.1)) +
  scale_x_continuous(limits = c(-3,3), breaks = c(-3,0,3), expand = c(0,0))
p
ggsave(p, file = "../output/figures/pilot.png", width = 5, height = 2, dpi = 600)

# 1. condition-level posterior means
cond_means <- draws %>%
  group_by(.draw, Response, conceptTypeL, isNegated) %>%
  summarise(mu = mean(.epred), .groups = "drop")

# 2. LC: absolute magnitude (distance from zero), then compare types
lc_table <- cond_means %>%
  filter(Response == "LC") %>%
  mutate(abs_mu = abs(mu)) %>%
  group_by(.draw, isNegated, conceptTypeL) %>%
  summarise(abs_mu = mean(abs_mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptTypeL, values_from = abs_mu) %>%
  mutate(diff = dishonest - honest) %>%
  group_by(Response = "LC", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

# 3. CWC: signed contrasts
cwc_table <- cond_means %>%
  filter(Response == "CWC") %>%
  group_by(.draw, isNegated, conceptTypeL) %>%
  summarise(mu = mean(mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptTypeL, values_from = mu) %>%
  mutate(diff = dishonest - honest) %>%
  group_by(Response = "CWC", isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

# 4. combine
summary_table <- bind_rows(lc_table, cwc_table)

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

