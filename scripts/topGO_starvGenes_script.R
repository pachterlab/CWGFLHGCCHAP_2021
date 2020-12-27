install.packages('rlang')
if (!requireNamespace("BiocManager", quietly=TRUE))
  + install.packages("BiocManager")
#BiocManager::install(version = "3.10") #Only for R 3.6

BiocManager::install("topGO")

library(topGO)
library(ALL)
library(readr)

#Read in DE genes (XLOC's) with GO Terms
geneID2GO <- readMappings(file = "~/Desktop/atlas_deseq2_genes_fortopGO.txt")
str(head(geneID2GO ))

#Add gene modules as factor 
atlas_deseq2_genes_fortopGO_metadata <- read_delim("~/Desktop/atlas_deseq2_genes_fortopGO_metadata.txt", 
                                                   "\t", escape_double = FALSE, col_names = FALSE, 
                                                   trim_ws = TRUE)

#Set variables
allMods = unique(atlas_deseq2_genes_fortopGO_metadata$X3)
alpha = 0.05/length(allMods) #Bonferroni correction, could correct for all pairwise comparisons?


getEnrichTerms <- function(geneID2GO, modMetadata, clus){
  mods <- factor(as.integer(modMetadata$X3 == clus)) #Choose gene module to make 'interesting'
  names(mods) <- names(geneID2GO)

  
  #Get genes only in module of interest
  clusGenes <- function(mods) {
    return(mods == 1)
  }
  subMods <- clusGenes(mods)
  
  #Make GO data
  GOdata <- new("topGOdata", ontology = "BP", allGenes = mods,
                geneSel = subMods, annot = annFUN.gene2GO, gene2GO = geneID2GO)
  
  #GOdata
  #sigGenes(GOdata)
  
  resultFis <- runTest(GOdata, algorithm = "classic", statistic = "fisher")
  
  resultWeight <- runTest(GOdata, statistic = "fisher")

  #P-values from Weight Algorithm
  pvalsWeight <- score(resultWeight)
  
  #hist(pvalsWeight, 50, xlab = "p-values")
  
  allRes <- GenTable(GOdata, classic = resultFis, weight = resultWeight, 
                     orderBy = "weight", ranksOf = "classic", topNodes = 20)
  
  subRes <- allRes[as.numeric(allRes$weight) < alpha,]
  
  #Write output
  write.csv(subRes,file=paste('mod',clus,'_GOTerms.csv',sep=""))
}

#Run for all modules and write outputs
for(c in allMods){
  
  getEnrichTerms(geneID2GO = geneID2GO,modMetadata = atlas_deseq2_genes_fortopGO_metadata, clus = c)
  
}

