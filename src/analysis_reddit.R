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
df <- read.csv("../output/scores/abstracted_scores_reddit.csv", header=T)
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

meta <- meta %>% filter(!(target_word == "cruel" & setPair == "virtuous_vicious"))
  
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
    LC = SC_signed,
    CWC = cw_SC_signed,
  ) %>% 
  mutate(across(c(targetWord, dependency, isNegated, conceptType, id, subreddit, setPair), ~ as.factor(.)))

any(is.na(df$conceptType))
any(is.na(df$subreddit))
any(is.na(df$dependency))
any(is.na(df$targetWord))
any(is.na(df$isNegated))
any(is.na(df$setPair))
table(df$isNegated)

df$targetWord %>% 
  table

df$setPair %>% 
  table

# Inspect for bimodality
ggplot(df, aes(x = LC, fill = conceptType)) +
  geom_histogram() +
  facet_grid(~isNegated, scales = "free_y")

ggplot(df, aes(x = CWC, fill = conceptType)) +
  geom_density() +
  facet_grid(~isNegated, scales = "free_y")

df %>%
  summarise(
    LC_zero = mean(LC == 0),
    CWC_zero = mean(CWC == 0)
  )

# Multivariate formula

#### Fully controlled corpus
bf_sc <- bf(
  LC ~ conceptType * isNegated +
    log1p(wordCount) +
    #(1 | setPair:targetWord) +
    #(1 + isNegated || setPair) +
    (1 | setPair) +
    (1 | dependency) +
    (1 | subreddit),
  family = gaussian()
)

bf_cwc <- bf(
  CWC ~ conceptType * isNegated +
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
  bf_sc + bf_cwc + set_rescor(TRUE),
  data = df,
  chains = 6, 
  cores = 6,
  threads = threading(2),
  iter = 2000,
  warmup = 1000,
  refresh = 50,
  seed = 3879,
  backend = "cmdstanr",
  file = "../output/models/r_eval.rds",
  control = list(adapt_delta = 0.99, max_treedepth = 12)
)

doParallel::stopImplicitCluster()
system("ps aux | grep cmdstan")
system("pkill -f cmdstan")

summary(fit)
bayes_R2(fit)

bf_sc <- bf(
  LC ~ conceptType * isNegated +
    conceptType * log1p(wordCount) +
    #(1 | setPair:targetWord) +
    #(1 + isNegated || setPair) +
    (1 | setPair) +
    (1 | dependency) +
    (1 | subreddit),
  family = gaussian()
)

bf_cwc <- bf(
  CWC ~ conceptType * isNegated +
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
  bf_sc + bf_cwc + set_rescor(TRUE),
  data = df,
  chains = 6, 
  cores = 6,
  threads = threading(2),
  iter = 2000,
  warmup = 1000,
  refresh = 50,
  seed = 3879,
  backend = "cmdstanr",
  file = "../output/models/r_eval_xWC.rds",
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
draws <- draws %>%
  rename(Response = .category)

draws <- draws %>% mutate(isNegated = ifelse(isNegated == "TRUE", "Under negation", "Affirmation"))

# 1. condition-level posterior means
cond_means <- draws %>%
  group_by(.draw, Response, conceptType, isNegated) %>%
  summarise(mu = mean(.epred), .groups = "drop")

# 2. LC: absolute magnitude (distance from zero), then compare types
lc_table <- cond_means %>%
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
cwc_table <- cond_means %>%
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
summary_table <- bind_rows(lc_table, cwc_table)
summary_table
summary_table %>% xtable(., digits = 4) %>% print(include.rownames = F)

plotdata <- draws %>% mutate(
  conceptType = ifelse(conceptType == "PTT", "Positive Thick", "Negative Thick")
)

p <- ggplot(plotdata, aes(x = .epred, y = Response, fill = conceptType)) +
  stat_halfeye(.width = c(0.66, 0.95), size = 2, alpha = 0.8) +
  scale_fill_manual(
    values = MetBrewer::met.brewer("Lakota", n = length(unique(plotdata$Response)))
  ) +
  theme_light() +
  theme(panel.spacing = unit(1, "lines"),
        plot.caption = element_markdown(hjust = 0, size = 7),
        strip.text.x = element_markdown(size = 10, margin = margin(0.2,0,0,0, "cm")),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 10),
        legend.position = "top"
  ) +
  facet_grid(~isNegated) +
  labs(fill = "Term Type",
       x = "Predicted Response",
       y = NULL,
       caption = paste0(
         "<br>bf((LC,CWC) ~ conceptType × isNegated + log1p(wordCount) + (1|setPair) + (1|dependency) + (1|subreddit), family = gaussian())"
       )) +
  scale_y_discrete(expand = c(0,0.1)) +
  scale_x_continuous(limits = c(-1.25,1.25), breaks = c(-1,0,1), expand = c(0,0))
p

ggsave(p, file = "../output/figures/reddit.png", width = 6, height = 3, dpi = 600)

ggplot(df, aes(x=LC, fill = conceptType)) +
  geom_density()

ggplot(df, aes(x=CWC, fill = conceptType)) +
  geom_density()


drawsl <- draws %>%
  mutate(divergence = abs(LC_epred - CWC_epred))
