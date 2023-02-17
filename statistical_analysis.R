# library("rrr")
##library(tidyverse)
library("Hmisc")
matching_res <- read.csv2("./formatted_results/validation_metrics_computed_with_metrics.csv", header = TRUE,sep=",", dec = ".")


FHIR_matching_res <- matching_res[which(matching_res$model_1=="FHIR" & matching_res$tool=="COMA"), names(matching_res) %in% c("F1.score","tool","model_1","model_2","model_1_size","model_2_size")]


ODATIS_matching_res <- matching_res[which(matching_res$model_1=="ODATIS" & matching_res$tool=="COMA"), names(matching_res) %in% c("F1.score","tool","model_1","model_2","model_1_size","model_2_size")]

# datas = data.frame(matching_time)
# head(datas, 5)
model <-lm(F1.score ~ model_1_size + model_2_size , data = matching_res)

summary(model)
# model <-lm(F1.score ~ model_1_size + model_2_size , data = ODATIS_matching_res)
#
# summary(model)
#
# model <-lm(F1.score ~ model_1_size + model_2_size , data = FHIR_matching_res)

# summary(model)
# # Call:
# # lm(formula = F1.score ~ model_1_size + model_2_size, data = matching_res)
# #
# # Residuals:
# #       Min        1Q    Median        3Q       Max
# # -0.082964 -0.015587 -0.001550  0.007249  0.131907
# #
# # Coefficients:
# #                Estimate Std. Error t value Pr(>|t|)
# # (Intercept)   8.532e-02  3.549e-03  24.038  < 2e-16 ***
# # model_1_size -5.536e-05  9.592e-06  -5.772 1.35e-08 ***
# # model_2_size -1.741e-05  1.995e-06  -8.725  < 2e-16 ***
# # ---
# # Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# #
# # Residual standard error: 0.04232 on 515 degrees of freedom
# # Multiple R-squared:  0.3491,	Adjusted R-squared:  0.3466
# # F-statistic: 138.1 on 2 and 515 DF,  p-value: < 2.2e-16
#
#
#
# model <-lm(F1.score ~ model_1_intermediate_node + model_2_intermediate_node , data = matching_res)
#
# summary(model)


# Call:
# lm(formula = F1.score ~ model_1_intermediate_node + model_2_intermediate_node,
#     data = matching_res)
#
# Residuals:
#       Min        1Q    Median        3Q       Max
# -0.073498 -0.018090 -0.003691  0.010176  0.140365
#
# Coefficients:
#                             Estimate Std. Error t value Pr(>|t|)
# (Intercept)                7.350e-02  4.317e-03  17.026   <2e-16 ***
# model_1_intermediate_node -1.359e-05  1.985e-05  -0.684    0.494
# model_2_intermediate_node -2.700e-05  1.904e-06 -14.182   <2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
#
# Residual standard error: 0.04401 on 515 degrees of freedom
# Multiple R-squared:  0.2958,	Adjusted R-squared:  0.2931
# F-statistic: 108.2 on 2 and 515 DF,  p-value: < 2.2e-16
#
#
# cor(matching_res$F1.score, matching_res$model_1_size*matching_res$model_2_size  )
# cor(as.matrix(matching_res),method = c("pearson", "kendall", "spearman"), use = "complete.obs")
# rcorr(as.matrix(matching_res))

# Syntax
library(corrplot)
no_empty <- matching_res[,!names(matching_res) %in% c("experiment","X","Unnamed..0","tool","model_1","model_2")]
# corrplot(no_empty)
rcorr_matr <- rcorr(as.matrix(no_empty), type = "spearman")

print(rcorr_matr)

# model <-lm(F1.score ~ model_1_average_word_frequency^2 + model_2_average_word_frequency^2 , data = matching_res)
# #
# summary(model)
# #
# model <-lm(F1.score ~ model_1_weighted_average_word_frequency^2 + model_2_weighted_average_word_frequency^2 , data = matching_res)
#
# summary(model)
# anova(model)
# plot(model)



