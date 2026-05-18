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

df %>%
  group_by(conceptType, isNegated) %>%
  summarise(median_wc = median(wordCount), .groups = "drop")

table(df$conceptType)


df %>%
  count(conceptType, isNegated) %>%
  group_by(conceptType) %>%
  mutate(prop = n / sum(n))
