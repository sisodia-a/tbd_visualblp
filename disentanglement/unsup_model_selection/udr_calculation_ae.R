## Christie's Data

rm(list=ls())

library(dplyr)
library(stringr)
library(ggplot2)
library(ggcorrplot)
library(cowplot)
library(stargazer)
# library(broom)
library(caret)
library(expm)

args <- commandArgs(trailingOnly = TRUE)
ARGUMENT <- args[1]

# beta <- 50
# m <- 0

delete_columns <- function(df) {
for(i in ncol(df):1)
{
  mean <- mean(df[,i])
  sd <- sd(df[,i])
  if(mean==0 & sd==0)
  {
    df[,i] <- NULL
  }
}
  return(df)
}

calculate_udr <- function(z_i,z_j) {
  df <- matrix(NA,ncol(z_j),ncol(z_i))
  for(i in 1:ncol(z_i))
  {
    for(j in 1:ncol(z_j))
    {
      df[j,i] <- cor(z_i[,i], z_j[,j],method="spearman")
    }
  }
  df <- abs(df)
#  df <- sqrtm(t(df) %*% df)
  udr_row <- 0
  for(i in 1:nrow(df))
  {
    udr_row <- udr_row + (max(df[i,]) * max(df[i,]))/sum(df[i,])
  }
  udr_col <- 0
  for(j in 1:ncol(df))
  {
    udr_col <- udr_col + (max(df[,j]) * max(df[,j]))/sum(df[,j])
  }
  udr_score <- (udr_row+udr_col)/(nrow(df)+ncol(df))
  # print(paste("UDR Score", udr_score))
  return(udr_score)
}

udr_beta_inner <- function(seed) {
  focal_df <- read.csv(paste0(ARGUMENT,"_s",seed,"_total.csv"))
  focal_df$file_name <- NULL
  focal_df <- delete_columns(focal_df)
  udr_inner = 0
  for(i in 1:10)
  {
    if(i!=seed)
    {
      comparison_df <- read.csv(paste0(ARGUMENT,"_s",i,"_total.csv"))
      comparison_df$file_name <- NULL
      comparison_df <- delete_columns(comparison_df)
      udr_inner <- udr_inner + calculate_udr(focal_df,comparison_df)
      }
  }
  # print(paste("Mean UDR Inner",udr_inner/9))
  return(udr_inner/9)
}

udr_beta_outer <- function()
{
  udr_outer <- 0
  for(i in 1:10)
  {
    udr_outer <- udr_outer + udr_beta_inner(i)
  }
  # print(paste("Mean UDR Outer",udr_outer/10))
  return((udr_outer)/10)
}

print(udr_beta_outer())

