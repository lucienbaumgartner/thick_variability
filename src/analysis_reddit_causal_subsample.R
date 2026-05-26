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
df <- read.csv("../output/scores/abstracted_scores_reddit_causal_sample.csv", header=T)
str(df)
df <- df %>% mutate(isNegated = ifelse(is_negated == "True", TRUE, FALSE))
df <- df %>% select(-is_negated)

# Get metadata
meta <- fromJSON("../input/target_terms.json") %>% as.data.frame
meta <- meta %>%
  unnest_longer(pos, values_to = "target_word") %>%
  mutate(polarity = "pos") %>%
  select(pair, polarity, target_word) %>%
  bind_rows(
    meta %>%
      unnest_longer(neg, values_to = "target_word") %>%
      mutate(polarity = "neg") %>%
      select(pair, polarity, target_word)
  )

meta <- meta %>% 
  rename(conceptType = polarity, setPair = pair) %>% 
  mutate(conceptType = ifelse(conceptType == "pos", "PTT", "NTT"))

#meta <- meta %>% filter(!(target_word == "cruel" & setPair == "virtuous_vicious"))

# Join metadata
df <- left_join(df, meta, by = "target_word")

# Join subreddit
reddit <- read.csv("../output/corpora/reddit/built_corpus/deduplicated_corpus.csv")
reddit <- reddit %>% select(id, subreddit)
reddit <- reddit[!duplicated(reddit),]

df <- left_join(df, reddit, by = "id")

# Create additional predictors
df <- df %>%
  rename(
    targetWord = target_word,
    wordCount = sentence_wordcount,
    CEE = cf_effect,
    RE = replacement_entropy,
  ) %>% 
  mutate(across(c(targetWord, dependency, isNegated, conceptType, id, subreddit, setPair), ~ as.factor(.)))

any(is.na(df$conceptType))
any(is.na(df$subreddit))
any(is.na(df$dependency))
any(is.na(df$targetWord))
any(is.na(df$isNegated))
any(is.na(df$setPair))
table(df$isNegated)

df <- df %>% filter(!is.na(CEE) & !is.na(RE))

df$targetWord %>% 
  table

df$setPair %>% 
  table

# Inspect for bimodality
ggplot(df, aes(x = CEE, fill = conceptType)) +
  geom_histogram() +
  facet_grid(~isNegated, scales = "free_y")

ggplot(df, aes(x = RE, fill = conceptType)) +
  geom_density() +
  facet_grid(~isNegated, scales = "free_y")

df %>%
  summarise(
    CEE_zero = mean(CEE == 0),
    RE_zero = mean(RE == 0)
  )

# Multivariate formula

#### Fully controlled corpus
bf_cee <- bf(
  CEE ~ conceptType * isNegated +
    log1p(wordCount) +
    #(1 | setPair:targetWord) +
    #(1 + isNegated || setPair) +
    (1 | setPair) +
    (1 | dependency) +
    (1 | subreddit),
  family = gaussian()
)

bf_re <- bf(
  RE ~ conceptType * isNegated +
    log1p(wordCount) +
    #(1 | setPair:targetWord) +
    #(1 + isNegated || setPair) +
    (1 | setPair) +
    (1 | dependency) +
    (1 | subreddit),
  family = gaussian()
)

parallel::detectCores()

fit <- brm(
  bf_cee + bf_re + set_rescor(TRUE),
  data = df,
  chains = 6, 
  cores = 6,
  threads = threading(2),
  iter = 2000,
  warmup = 1000,
  refresh = 50,
  seed = 3879,
  backend = "cmdstanr",
  file = "../output/models/r_cf_ent.rds",
  control = list(adapt_delta = 0.99, max_treedepth = 12)
)

doParallel::stopImplicitCluster()
system("ps aux | grep cmdstan")
system("pkill -f cmdstan")

summary(fit)
bayes_R2(fit)

bf_cee <- bf(
  CEE ~ conceptType * isNegated +
    conceptType * log1p(wordCount) +
    #(1 | setPair:targetWord) +
    #(1 + isNegated || setPair) +
    (1 | setPair) +
    (1 | dependency) +
    (1 | subreddit),
  family = gaussian()
)

bf_re <- bf(
  RE ~ conceptType * isNegated +
    conceptType * log1p(wordCount) +
    #(1 | setPair:targetWord) +
    #(1 + isNegated || setPair) +
    (1 | setPair) +
    (1 | dependency) +
    (1 | subreddit),
  family = gaussian()
)

parallel::detectCores()

fit2 <- brm(
  bf_cee + bf_re + set_rescor(TRUE),
  data = df,
  chains = 6, 
  cores = 6,
  threads = threading(2),
  iter = 2000,
  warmup = 1000,
  refresh = 50,
  seed = 3879,
  backend = "cmdstanr",
  file = "../output/models/r_cf_ent_xWC.rds",
  control = list(adapt_delta = 0.99, max_treedepth = 12)
)

doParallel::stopImplicitCluster()
system("ps aux | grep cmdstan")
system("pkill -f cmdstan")

loo1 <- loo(fit)
loo2 <- loo(fit2)
loo_compare(loo1, loo2)


# Posterior draws of NEC and SC for each combination of conceptType and isNegated
# Define all combinations of conceptType and isNegated
#wc_quantiles <- quantile(df$wordCount, probs = c(0.25, 0.50, 0.75))

newdata <- expand.grid(
  conceptType = c("PTT", "NTT"),
  isNegated   = c("FALSE", "TRUE"),
  wordCount   = median(df$wordCount)
)

# Get posterior draws
draws <- fit %>%
  epred_draws(newdata = newdata, re_formula = NA)  # Use re_formula = NA to ignore any random effects (none here)

draws <- draws %>% mutate(isNegated = ifelse(isNegated == "TRUE", "Under negation", "Affirmation"))

# 1. condition-level posterior means
cond_means <- draws %>%
  group_by(.draw, .category, conceptType, isNegated) %>%
  summarise(mu = mean(.epred), .groups = "drop")

# 2. CEE: absolute magnitude (distance from zero), then compare types
res <- cond_means %>%
  mutate(mu = abs(mu)) %>%
  group_by(.draw, .category, isNegated, conceptType) %>%
  summarise(mu = mean(mu), .groups = "drop") %>%
  pivot_wider(names_from = conceptType, values_from = mu) %>%
  mutate(diff = NTT - PTT) %>%
  group_by(.category, isNegated) %>%
  summarise(
    mean_diff = mean(diff),
    lo = quantile(diff, .025),
    hi = quantile(diff, .975),
    p_gt_0 = mean(diff > 0),
    .groups = "drop"
  )

res %>% xtable(., digits = 4) %>% print(include.rownames = F)

plotdata <- draws %>% mutate(
  conceptType = ifelse(conceptType == "PTT", "Positive Thick", "Negative Thick"),
  isNegated = factor(str_to_title(isNegated), levels = c("Under Negation", "Affirmation")),
  .category = ifelse(.category == "CEE", "Contextual Causal Effect", "Replacement Entropy")
)

p <- ggplot(plotdata, aes(x = .epred, y = isNegated, fill = conceptType)) +
  stat_halfeye(.width = c(0.66, 0.95), size = 2, alpha = 0.8) +
  scale_fill_manual(
    values = MetBrewer::met.brewer("Lakota",
                                   n = length(unique(plotdata$conceptType)))
  ) +
  facet_grid(~ .category, scales = "free_x") +
  
  ggh4x::facetted_pos_scales(
    x = list(
      .category == "Contextual Causal Effect"  ~ scale_x_continuous(
        limits = c(-1, 1),
        breaks = c(-1, 0, 1),
        expand = c(0, 0.1)
      ),
      .category == "Replacement Entropy" ~ scale_x_continuous(
        limits = c(2, 4),
        breaks = 2:4,
        expand = c(0, 0.1)
      )
    )
  ) +
  
  scale_y_discrete(expand = c(0, 0.1)) +
  theme_light() +
  theme(
    panel.spacing = unit(1, "lines"),
    plot.caption = element_markdown(hjust = 0, size = 6),
    strip.text.y = element_markdown(size = 10, margin = margin(0,0.1,0,0, "cm")),
    strip.text.x = element_markdown(size = 10),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 10),
    legend.position = "top",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 10)
  ) +
  labs(
    fill = "Term Type",
    x = "Predicted Response",
    y = NULL,
    caption = paste0(
      "<br>bf((CEE,RE) ~ conceptType × isNegated + log1p(wordCount) + (1|setPair) + (1|dependency) + (1|subreddit), family = gaussian())"
    ))
p
ggsave(p, file = "../output/figures/reddit_cf_ent_subsample.png", width = 6, height = 3, dpi = 600)

ggplot(df, aes(x=CEE, fill = conceptType)) +
  geom_density()

ggplot(df, aes(x=RE, fill = conceptType)) +
  geom_density()


drawsl <- draws %>%
  mutate(divergence = abs(CEE_epred - RE_epred))
