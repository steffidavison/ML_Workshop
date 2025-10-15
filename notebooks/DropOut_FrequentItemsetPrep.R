
# Environment -------------------------------------------------------------

library(dplyr)
library(tidyr)
library(readr)
library(magrittr)
library(ggplot2)
library(ggthemes)
library(santoku)
library(janitor)

set.seed(42)  # reproducibility

# Data --------------------------------------------------------------------

df <- read.csv('../data/DropOut.csv') %>% janitor::clean_names() %>% dplyr::select(-x)

dfSummary <- df %>%
    drop_na(calculated_od600_to_nanodrop) %>%
    mutate(residual_glucose_g_l = tidyr::replace_na(residual_glucose_g_l, -99.0)) %>%
    group_by(strain, media, type_of_doe, plate, incubation_time_h, condition, p_h, glucose_g_l,
    ammonium_sulfate_g_l, phosphate_citrate_x, ynb_x, amino_acid_1_m_m, amino_acid_2_g_l,
    amino_acid_2_g_l_1, ethanol_g_l, agit_rpm, amino_acid_mix_x) %>%
        summarise(mean_calc_OD60 = mean(calculated_od600_to_nanodrop),
        residual_glucose = max(residual_glucose_g_l)) %>%
    tibble()

dfSummary$ID <- seq(nrow(dfSummary))

# Data Discretization -----------------------------------------------------

## strain

strain <- dfSummary %>%
  dplyr::select(ID, strain) %>%
  na.omit() %>%
  dplyr::rename(Bin = strain)

## type_of_doe

type_of_doe <- dfSummary %>%
  dplyr::select(ID, type_of_doe) %>%
  na.omit() %>%
  dplyr::rename(Bin = type_of_doe)

## plate

plate <- dfSummary %>%
  dplyr::select(ID, plate) %>%
  na.omit() %>%
  dplyr::rename(Bin = plate)

## condition

condition <- dfSummary %>%
  dplyr::select(ID, condition) %>%
  mutate(condition = as.factor(condition)) %>%
  na.omit() %>%
  dplyr::rename(Bin = condition)

## pH

pH <- dfSummary %>%
  dplyr::select(ID, p_h) %>%
  mutate(pH = as.factor(paste0('pH_', as.character(p_h)))) %>%
  dplyr::select(-p_h) %>%
  na.omit() %>%
  dplyr::rename(Bin = pH)

## glucose

glucose <- dfSummary %>%
  dplyr::select(ID, glucose_g_l) %>%
  mutate(glucose = as.factor(paste0('glucose_', as.character(glucose_g_l)))) %>%
  dplyr::select(-glucose_g_l) %>%
  na.omit() %>%
  dplyr::rename(Bin = glucose)

## ammoniumSulfate

ammoniumSulfate <- dfSummary %>%
  dplyr::select(ID, ammonium_sulfate_g_l) %>%
  mutate(ammoniumSulfate = as.factor(paste0('ammoniumSulfate_', as.character(ammonium_sulfate_g_l)))) %>%
  dplyr::select(-ammonium_sulfate_g_l) %>%
  na.omit() %>%
  dplyr::rename(Bin = ammoniumSulfate)

## phosphateCitrate

phosphateCitrate <- dfSummary %>%
  dplyr::select(ID, phosphate_citrate_x) %>%
  mutate(phosphateCitrate = as.factor(paste0('phosphateCitrate_', as.character(phosphate_citrate_x)))) %>%
  dplyr::select(-phosphate_citrate_x) %>%
  na.omit() %>%
  dplyr::rename(Bin = phosphateCitrate)

## ynb_x

ynbx <- dfSummary %>%
  dplyr::select(ID, ynb_x) %>%
  mutate(ynbx = as.factor(paste0('ynbx_', as.character(ynb_x)))) %>%
  dplyr::select(-ynb_x) %>%
  na.omit() %>%
  dplyr::rename(Bin = ynbx)

## amino_acid_1_m_m

aminoAcid1 <- dfSummary %>%
  dplyr::select(ID, amino_acid_1_m_m) %>%
  mutate(aminoAcid1 = as.factor(paste0('aminoAcid1_', as.character(amino_acid_1_m_m)))) %>%
  dplyr::select(-amino_acid_1_m_m) %>%
  na.omit() %>%
  dplyr::rename(Bin = aminoAcid1)

## amino_acid_2_g_l

aminoAcid2 <- dfSummary %>%
  dplyr::select(ID, amino_acid_2_g_l) %>%
  mutate(aminoAcid2 = as.factor(paste0('aminoAcid2_', as.character(amino_acid_2_g_l)))) %>%
  dplyr::select(-amino_acid_2_g_l) %>%
  na.omit() %>%
  dplyr::rename(Bin = aminoAcid2)

## amino_acid_2_g_l_1

aminoAcid2_1 <- dfSummary %>%
  dplyr::select(ID, amino_acid_2_g_l_1) %>%
  mutate(aminoAcid2_1 = as.factor(paste0('aminoAcid2_1_', as.character(amino_acid_2_g_l_1)))) %>%
  dplyr::select(-amino_acid_2_g_l_1) %>%
  na.omit() %>%
  dplyr::rename(Bin = aminoAcid2_1)

## ethanol_g_l

ethanol <- dfSummary %>%
  dplyr::select(ID, ethanol_g_l) %>%
  mutate(ethanol = as.factor(paste0('ethanol_', as.character(ethanol_g_l)))) %>%
  dplyr::select(-ethanol_g_l) %>%
  na.omit() %>%
  dplyr::rename(Bin = ethanol)

## residual_glucose

residualGlucose <- dfSummary %>%
  dplyr::select(ID, residual_glucose) %>%
  dplyr::filter(residual_glucose > -95.0) %>%
  na.omit() %>%
  mutate(Bin = as.character(chop(residual_glucose, c(5), labels = c('residualGlucose_LT5', 'residualGlucose_GT5')))) %>%
  dplyr::select(-residual_glucose)

## mean_calc_OD60

mean_calc_OD60 <- dfSummary %>%
  dplyr::select(ID, mean_calc_OD60) %>%
  na.omit() %>%
  mutate(Bin = as.character(chop(mean_calc_OD60, c(10), labels = c('OD_LT10', 'OD_GT10')))) %>%
  dplyr::select(-mean_calc_OD60)

# Write transactions ------------------------------------------------------

transactions <-
  rbind(strain,
        type_of_doe,
        plate,
        condition,
        pH,
        glucose,
        ammoniumSulfate,
        phosphateCitrate,
        ynbx,
        aminoAcid1,
        aminoAcid2,
        aminoAcid2_1,
        ethanol,
        residualGlucose,
        mean_calc_OD60)

write.csv(transactions, 'DropOutTransactions.csv')
