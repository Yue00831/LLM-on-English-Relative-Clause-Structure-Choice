#####################################################################################
##    Inspired by        ##
##    Erlewine&Kotek (2013)   ##
##    May 2013 Hadas Kotek, licensed under the MIT license  ##
#####################################################################################
#install.packages("plyr") # you don't need to install pkg every single time
library(plyr)
#install.packages("dplyr") # you need to type Yes into the console
library(dplyr)
#install.packages("stringr")
library(stringr)
#install.packages("lattice")
library(lattice)
library(ggplot2)
library(viridis)
# intercept-only mixed logistic regression in order to test for difference from 50% chance level
#install.packages("lme4")
library(lme4)
library(emmeans) #need when run post-hoc test

# read input file
setwd(dirname(rstudioapi::getSourceEditorContext()$path))
getwd()
rm(list=ls())
data <- read_excel("4LLM.xlsx")

#store animacy as factor instead of numerical
data$animacy <- factor(data$animacy)
# data$animacy <- relevel(data$animacy, ref = "1") ## If want to adjust the reference level

#Binomial on structure choice
model <- glmer(gpt2_choose_psv ~ factor(animacy) + (1|verb), data = data, family = 'binomial')
summary(model)

#poast-hoc:
emmeans_object <- emmeans(model, pairwise ~ animacy, adjust = "tukey")
summary(emmeans_object$contrasts)


#Is choosing_passive above chance level?
model_chancelevel = glmer(bertlarge_uncased_choose_psv ~ 1 + (1|verb) + (1|item), data = data, family='binomial')
summary(model_chancelevel)
