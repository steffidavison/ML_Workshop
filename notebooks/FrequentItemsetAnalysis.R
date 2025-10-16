# Packages ----------------------------------------------------------------

library(arules)
library(arulesViz)
library(tidyverse)
library(readxl)
library(knitr)
library(ggplot2)
library(lubridate)
library(plyr)
library(RColorBrewer)
# library(revealjs)

set.seed(42)

# read data ---------------------------------------------------------------

tr <-
  read.transactions(
    'DropOutTransactions.csv',
    format = 'single',
    header = TRUE,
    sep = ',',
    cols = c("ID", "Bin")
  )

# RHS ---------------------------------------------------------------------

RHS.association.rules <-
  apriori(
    tr,
    parameter = list(
      supp = 0.06,
      conf = 0.8,
      maxlen = 8,
      maxtime = 0
    )
    , appearance = list(default = "lhs", rhs = "OD_GT10")
  )

rules <- sort(RHS.association.rules, by = "confidence")
quality(rules)$improvement <- interestMeasure(rules, measure = "importance")

non_redundant_rules <- rules[!is.redundant(rules)]

maximal_rules <- rules[is.maximal(non_redundant_rules)]

arules::write(maximal_rules, 'DropOut_OD_GT10maximal_rules.csv', sep = ',')

plot(maximal_rules, method = "graph", engine = "html")

inspectDT(maximal_rules)

# RHS ---------------------------------------------------------------------

RHS.association.rules <-
  apriori(
    tr,
    parameter = list(
      supp = 0.06,
      conf = 1.0,
      maxlen = 8,
      maxtime = 0
    )
    , appearance = list(default = "lhs", rhs = "OD_LT10")
  )

rules <- sort(RHS.association.rules, by = "confidence")
quality(rules)$improvement <- interestMeasure(rules, measure = "importance")

non_redundant_rules <- rules[!is.redundant(rules)]

maximal_rules <- rules[is.maximal(non_redundant_rules)]

arules::write(maximal_rules, 'DropOut_OD_LT10maximal_rules.csv', sep = ',')

plot(maximal_rules[1:20], method = "graph", engine = "html")

inspectDT(maximal_rules)
