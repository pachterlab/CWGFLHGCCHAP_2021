library(usedist)
library(phangorn)
library(readr)
library(tidyverse)
library(hash)

splitsTree_df <-  as.tibble(read_csv("~/Downloads/geneClustDist_SplitsTree.csv"))
#splitsTree_df <-  as.tibble(read_csv("~/Downloads/cellTypeDist_SplitsTree (2).csv"))


#Make annotations Nexus compatible

#Change names
j = 0
na_j = 0
for (i in splitsTree_df["annos"][[1]]){
  j = j+1
  if (is.na(i)){
    na_j = na_j +1
    
    splitsTree_df["annos"][[1]][j] = paste("NA",as.character(na_j),sep="_")
    
  }
  
}

newName <- function(name){
  new = gsub("- ", "",name)
  new = gsub(" ", "_",new)
}

splitsTree_df[c("annos")] <- 
  lapply(splitsTree_df[c("annos")], function(col) map(col,newName))


#Convert to df for distance calculations
splitsTree_df <- as.data.frame(splitsTree_df)
rownames(splitsTree_df) <- splitsTree_df$annos
splitsTree_df <- subset(splitsTree_df,select=-c(annos,cluster,X1))


#Get pairwise (Euclidean) distances
d <- dist(splitsTree_df,upper = FALSE,diag = TRUE)

#Write distance matrix as nexus file

write.nexus.dist(d,file="~/Desktop/distGeneClus.nex", append = FALSE, upper = FALSE,
                 diag = TRUE, digits = getOption("digits"))