
<a href="https://colab.research.google.com/github/pachterlab/CWGFLHGCCHAP_2021/blob/master/notebooks/StimulationAnalysis/deSeq2Analysis_StimulationResponse.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
!date
```

### **Download Data**


```
import requests
from tqdm import tnrange, tqdm_notebook
def download_file(doi,ext):
    url = 'https://api.datacite.org/dois/'+doi+'/media'
    r = requests.get(url).json()
    netcdf_url = r['data'][0]['attributes']['url']
    r = requests.get(netcdf_url,stream=True)
    #Set file name
    fname = doi.split('/')[-1]+ext
    #Download file with progress bar
    if r.status_code == 403:
        print("File Unavailable")
    if 'content-length' not in r.headers:
        print("Did not get file")
    else:
        with open(fname, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            pbar = tnrange(int(total_length/1024), unit="B")
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    pbar.update()
                    f.write(chunk)
        return fname
```


```
#Import raw, unclustered stimulation data
download_file('10.22002/D1.1814','.gz')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=78502.0), HTML(value='')))





    'D1.1814.gz'




```
#Import previously saved, clustered, & filtered stimulation data 
download_file('10.22002/D1.1821','.gz')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=772581.0), HTML(value='')))





    'D1.1821.gz'




```
#Import merged data with knn clusters
download_file('10.22002/D1.1823','.gz')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=257856.0), HTML(value='')))





    'D1.1823.gz'




```
#Human ortholog annotations
download_file('10.22002/D1.1819','.gz')

#Panther annotations
download_file('10.22002/D1.1820','.gz')

#GO Terms
download_file('10.22002/D1.1822','.gz')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=528.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=515.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=227.0), HTML(value='')))





    'D1.1822.gz'




```
#Previously saved DeSeq2 results (perturbed genes in DI and KCl conditions)
download_file('10.22002/D1.1818','.gz')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=321.0), HTML(value='')))





    'D1.1818.gz'




```
!gunzip *.gz
```


```
!pip install --quiet anndata
!pip install --quiet scanpy
!pip install --quiet louvain

```

    [K     |████████████████████████████████| 122kB 5.6MB/s 
    [K     |████████████████████████████████| 10.2MB 5.6MB/s 
    [K     |████████████████████████████████| 71kB 6.2MB/s 
    [K     |████████████████████████████████| 51kB 5.1MB/s 
    [?25h  Building wheel for sinfo (setup.py) ... [?25l[?25hdone
    [K     |████████████████████████████████| 2.2MB 4.1MB/s 
    [K     |████████████████████████████████| 3.2MB 31.3MB/s 
    [?25h


```
!pip3 install --quiet rpy2
```

### **Import Packages**


```
import scanpy as sc
import anndata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy.sparse
import scipy.io as sio
import seaborn as sns
import random

import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import (KNeighborsClassifier,NeighborhoodComponentsAnalysis)
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE

sc.set_figure_params(dpi=125)
%load_ext rpy2.ipython
```


```
#Read in annotations
from io import StringIO

hg_ortho_df = pd.read_csv(StringIO(''.join(l.replace('|', '\t') for l in open('D1.1819'))),
            sep="\t",header=None,skiprows=[0,1,2,3])

hg_ortho_df[['XLOC','TCONS']] = hg_ortho_df[13].str.split(expand=True) 
hg_ortho_df[['Gene','gi']] = hg_ortho_df[3].str.split(expand=True) 
hg_ortho_df['Description']= hg_ortho_df[11]


panther_df = pd.read_csv('D1.1820',
            sep="\t",header=None) #skiprows=[0,1,2,3]



goTerm_df = pd.read_csv('D1.1822',
            sep=" ",header=None) #skiprows=[0,1,2,3]

```

### **Run DeSeq2 Analysis for Stimulation Perturbations (KCL and DI water)**

Set up data


```
#Remove clusters with < 10 cells per condition
bus_stim = anndata.read("D1.1814")

#Read in previously saved/clustered data with low count cells removed
bus_stim_clus = anndata.read("D1.1821")

bus_stim = bus_stim[bus_stim_clus.obs_names,]
#bus_fs_raw.obs['orgID'] = bus_fs_clus.obs['orgID']
bus_stim.obs['condition'] = bus_stim_clus.obs['condition']
bus_stim.obs['cellRanger_louvain'] = bus_stim_clus.obs['cellRanger_louvain']
bus_stim


#clusSize
```

    Trying to set attribute `.obs` of view, copying.





    AnnData object with n_obs × n_vars = 18921 × 46716
        obs: 'batch', 'condition', 'cellRanger_louvain'




```
def clusToKeep(bus_fs_clus):
  keep = []
  clusSize = {}
  for i in np.unique(bus_fs_clus.obs['cellRanger_louvain']):
      cells = bus_fs_clus[bus_fs_clus.obs['cellRanger_louvain'].isin([i])]
      kcl_cells = len(cells[cells.obs['condition']=='KCl'].obs_names)
      di_cells = len(cells[cells.obs['condition']=='DI'].obs_names)
      sw_cells = len(cells[cells.obs['condition']=='SW'].obs_names)
      min_cells = np.min([kcl_cells,di_cells,sw_cells])
      if min_cells > 10:
        keep += [i]
        #clusSize[i] = min_cells
  return keep
```


```
#Subsample from full dataset, across each cluster
def getSampled_Cluster(bus_fs_clus,bus_fs_raw,keep):

  subSample = 150 #100
  cellNames = np.array(bus_fs_clus.obs_names)
  kcl = np.array(list(bus_fs_clus.obs['condition'] == 'KCl'))
  di = np.array(list(bus_fs_clus.obs['condition'] == 'DI'))
  sw = np.array(list(bus_fs_clus.obs['condition'] == 'SW'))

  allCells = []
  for i in keep:
      #subSample =  clusSize[i] 
      
      cells = np.array(list(bus_fs_clus.obs['cellRanger_louvain'].isin([i])))
      kcl_cells = list(np.where(kcl & cells)[0])
      di_cells = list(np.where(di & cells)[0])
      sw_cells = list(np.where(sw & cells)[0])
      
      #Take all cells if < subSample
      if len(kcl_cells) >= subSample:
          kcl_choice = random.sample(kcl_cells,subSample)
      else:
          kcl_choice = kcl_cells

      if len(di_cells) >= subSample:
          di_choice = random.sample(di_cells,subSample)
      else:
          di_choice = di_cells
          
      if len(sw_cells) >= subSample:
          sw_choice = random.sample(sw_cells,subSample)
      else:
          sw_choice = sw_cells
          
          
      pos = list(kcl_choice)+list(di_choice)+list(sw_choice)
      #print(len(pos))
      
      allCells += list(cellNames[pos])

      
  sub_raw = bus_fs_raw[allCells,:]
  return sub_raw
```

Create subsampled dataset of cells x genes (subsamples across cell types)


```
#For full dataset don't filter by highly variable
keep = clusToKeep(bus_stim_clus)
sub_raw = getSampled_Cluster(bus_stim_clus,bus_stim,keep)


sub_raw_copy = sub_raw.copy()

sc.pp.filter_cells(sub_raw, min_counts=1)
sc.pp.filter_genes(sub_raw, min_counts=1)

sub_raw

# sc.pp.normalize_per_cell(sub_raw_copy, counts_per_cell_after=1e4)
# sub_raw_copy.raw = sc.pp.log1p(sub_raw_copy, copy=True)

# sc.pp.highly_variable_genes(sub_raw_copy,n_top_genes=5000) #This is just a small example, for full data used all nonzero genes
# sub_raw = sub_raw[:,sub_raw_copy.var['highly_variable']]

```

    Trying to set attribute `.obs` of view, copying.





    AnnData object with n_obs × n_vars = 10887 × 37881
        obs: 'batch', 'condition', 'cellRanger_louvain', 'n_counts'
        var: 'n_counts'




```
#Instantiate dataframe with gene names, convert to R-compatible format
def makeDF_forR(sub_raw):
  fullDF = pd.DataFrame(scipy.sparse.csr_matrix.toarray(sub_raw.X).T, index = sub_raw.var_names.tolist(), columns= sub_raw.obs_names.tolist())
  conds = sub_raw.obs['condition'].tolist()
  #ids = sub_jelly4Raw.obs['orgID'].tolist()
  clus = sub_raw.obs['cellRanger_louvain'].tolist()

  reps = np.repeat(0,len(sub_raw.obs_names))

  length = len(sub_raw[sub_raw.obs['condition'] == 'KCl'].obs_names)
  reps[sub_raw.obs['condition'] == 'KCl'] = range(1,length+1)

  length = len(sub_raw[sub_raw.obs['condition'] == 'DI'].obs_names)
  reps[sub_raw.obs['condition'] == 'DI'] = range(1,length+1)

  length = len(sub_raw[sub_raw.obs['condition'] == 'SW'].obs_names)
  reps[sub_raw.obs['condition'] == 'SW'] = range(1,length+1)


  sampleDF = pd.DataFrame({'cell_ID': fullDF.columns}) \
          .assign(condition = conds) \
          .assign(replicate = reps) \
          .assign(cluster = clus) 
  sampleDF.index = sampleDF.cell_ID
  sampleDF.head()

  fullDF.to_csv('fullDF.csv')
  sampleDF.to_csv('sampleDF.csv')
```

##### **Read data into R and run DeSeq2 Models**


```
makeDF_forR(sub_raw)
```


```r
%%R 
fullDF <- read.csv(file = 'fullDF.csv')
sampleDF <- read.csv(file = 'sampleDF.csv')
head(sampleDF)
```

                 cell_ID          cell_ID.1 condition replicate cluster
    1 TTGCCTGAGATCCCAT-2 TTGCCTGAGATCCCAT-2       KCl         1       0
    2 CTCTGGTCATATGCGT-2 CTCTGGTCATATGCGT-2       KCl         2       0
    3 CTATAGGTCGCCAGAC-1 CTATAGGTCGCCAGAC-1       KCl         3       0
    4 CGAGGAACACGCTGCA-2 CGAGGAACACGCTGCA-2       KCl         4       0
    5 CTCAGAACACGGGTAA-2 CTCAGAACACGGGTAA-2       KCl         5       0
    6 GGGTGTCTCACCATCC-2 GGGTGTCTCACCATCC-2       KCl         6       0



```r
%%R
rownames(sampleDF) <- sampleDF$cell_ID 
#Replace '.' in cell barcodes with '-'
rownames(fullDF) <- fullDF$X
colnames(fullDF) <- gsub("\\.", "-", colnames(fullDF))
fullDF <- subset(fullDF, select = -c(X) )
#head(fullDF)

sampleDF <- subset(sampleDF, select = -c(cell_ID.1) )
head(sampleDF)
sampleDF$condition <- factor(sampleDF$condition)
```


```r
%%R
head(sampleDF)
```

                                  cell_ID condition replicate cluster
    TTGCCTGAGATCCCAT-2 TTGCCTGAGATCCCAT-2       KCl         1       0
    CTCTGGTCATATGCGT-2 CTCTGGTCATATGCGT-2       KCl         2       0
    CTATAGGTCGCCAGAC-1 CTATAGGTCGCCAGAC-1       KCl         3       0
    CGAGGAACACGCTGCA-2 CGAGGAACACGCTGCA-2       KCl         4       0
    CTCAGAACACGGGTAA-2 CTCAGAACACGGGTAA-2       KCl         5       0
    GGGTGTCTCACCATCC-2 GGGTGTCTCACCATCC-2       KCl         6       0



```r
%%R
install.packages("BiocManager")
BiocManager::install(version = "3.10")

```

    R[write to console]: Installing package into ‘/usr/local/lib/R/site-library’
    (as ‘lib’ is unspecified)
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/BiocManager_1.30.10.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 40205 bytes (39 KB)
    
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: 
    
    R[write to console]: downloaded 39 KB
    
    
    R[write to console]: 
    
    R[write to console]: 
    R[write to console]: The downloaded source packages are in
    	‘/tmp/RtmpSf6geO/downloaded_packages’
    R[write to console]: 
    R[write to console]: 
    
    R[write to console]: Error: Bioconductor version '3.10' requires R version '3.6'; see
      https://bioconductor.org/install
    


    
    Error: Bioconductor version '3.10' requires R version '3.6'; see
      https://bioconductor.org/install



```
!sudo apt-get update
!sudo apt-get install libxml2-dev
!sudo apt-get install r-cran-xml
!sudo apt-get install libcurl4-openssl-dev
```

    Ign:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
    Hit:2 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ InRelease
    Hit:3 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic InRelease
    Hit:4 http://archive.ubuntu.com/ubuntu bionic InRelease
    Get:5 http://security.ubuntu.com/ubuntu bionic-security InRelease [88.7 kB]
    Ign:6 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
    Hit:7 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release
    Hit:8 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
    Get:9 http://archive.ubuntu.com/ubuntu bionic-updates InRelease [88.7 kB]
    Hit:10 http://ppa.launchpad.net/cran/libgit2/ubuntu bionic InRelease
    Hit:11 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
    Get:12 http://archive.ubuntu.com/ubuntu bionic-backports InRelease [74.6 kB]
    Fetched 252 kB in 1s (199 kB/s)
    Reading package lists... Done
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    libxml2-dev is already the newest version (2.9.4+dfsg1-6.1ubuntu1.3).
    0 upgraded, 0 newly installed, 0 to remove and 62 not upgraded.
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    r-cran-xml is already the newest version (3.99-0.5-1cran1.1804.0).
    0 upgraded, 0 newly installed, 0 to remove and 62 not upgraded.
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    libcurl4-openssl-dev is already the newest version (7.58.0-2ubuntu3.12).
    0 upgraded, 0 newly installed, 0 to remove and 62 not upgraded.



```r
%%R 
#install.packages("DESeq2",repos = "http://cran.us.r-project.org")
BiocManager::install("DESeq2")
```

    R[write to console]: Bioconductor version 3.12 (BiocManager 1.30.10), R 4.0.3 (2020-10-10)
    
    R[write to console]: Installing package(s) 'DESeq2'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 2066943 bytes (2.0 MB)
    
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: 
    
    R[write to console]: downloaded 2.0 MB
    
    
    R[write to console]: 
    
    R[write to console]: 
    R[write to console]: The downloaded source packages are in
    	‘/tmp/RtmpSf6geO/downloaded_packages’
    R[write to console]: 
    R[write to console]: 
    
    R[write to console]: Old packages: 'BH', 'cpp11', 'crosstalk', 'DBI', 'diffobj', 'dplyr', 'DT',
      'fansi', 'gdtools', 'hms', 'htmltools', 'Rcpp', 'tibble', 'withr', 'xfun',
      'Matrix'
    



```
#Make output directory
!mkdir kallistoDEAnalysis_Stim
```

    mkdir: cannot create directory ‘kallistoDEAnalysis_Stim’: File exists



```r
%%R 
install.packages("DESeq2",repos = "http://cran.us.r-project.org")
library("DESeq2")
#library("apeglm")
#library(Rcpp)
#.libPaths()

clusters <- unique(sampleDF$cluster)
Genes <- c()
Cluster <- c()
Condition <- c() 
padj <- c()
log2FC <- c()

for (i in clusters){
 
        indices = which(sampleDF$cluster == i)
        subset = fullDF[,indices]
        subset_meta = subset(sampleDF,cluster == i)


        dds <- DESeqDataSetFromMatrix(countData = subset, colData = subset_meta, design= ~replicate + condition)

        #Set control condition
        dds$condition <- relevel(dds$condition, ref = 'SW')
        dds <- DESeq(dds,test="LRT", reduced=~replicate, sfType="poscounts", useT=TRUE, minmu=1e-6, 
                     minReplicatesForReplace=Inf,betaPrior = FALSE)#parallel = TRUE

        fc = 1 #usually fold change cutoff
        #SW v KCl results
        res <- results(dds,alpha=0.05,name="condition_KCl_vs_SW")
        resLFC <- res 

        resLFC <- na.omit(resLFC)
        resOrdered <- resLFC[resLFC$padj < .05,]
        #Keep log2 fold changes < -1 or > 1
        resOrdered <- resOrdered[abs(resOrdered$log2FoldChange) > fc,]
        outcomes <- resOrdered[order(resOrdered$padj),]

        Genes <- c(Genes,row.names(outcomes))
        Cluster <- c(Cluster,rep(i,length(row.names(outcomes))))
        Condition <- c(Condition,rep('KCl',length(row.names(outcomes)))) 
        padj <- c(padj,outcomes$padj)
        log2FC <- c(log2FC,outcomes$log2FoldChange)


        #SW v DI results
        res <- results(dds,alpha=0.05,name="condition_DI_vs_SW")
        resLFC <- res 

        resLFC <- na.omit(resLFC)
        resOrdered <- resLFC[resLFC$padj < .05,]
        #Keep log2 fold changes < -1 or > 1
        resOrdered <- resOrdered[abs(resOrdered$log2FoldChange) > fc,] 
        outcomes <- resOrdered[order(resOrdered$padj),]

        Genes <- c(Genes,row.names(outcomes))
        Cluster <- c(Cluster,rep(i,length(row.names(outcomes))))
        Condition <- c(Condition,rep('DI',length(row.names(outcomes)))) 
        padj <- c(padj,outcomes$padj)
        log2FC <- c(log2FC,outcomes$log2FoldChange)
         
    
}

deGenesDF <- data.frame(matrix(ncol = 6, nrow = length(Genes)))
names(deGenesDF) <- c("Genes", "Cluster", "Condition","padj","padjClus","log2FC")

deGenesDF$Genes <- Genes
deGenesDF$Cluster <- Cluster
deGenesDF$Condition <- Condition
deGenesDF$padj <- padj
deGenesDF$padjClus <- padj*length(unique(Cluster))
deGenesDF$log2FC <- log2FC

write.csv(deGenesDF,'./kallistoDEAnalysis_Stim/deSeq2_deGenesDF_log2FCof1_singleCellReplicates_noShrinkage_subSample.csv')

head(deGenesDF)
```

    R[write to console]: Installing package into ‘/usr/local/lib/R/site-library’
    (as ‘lib’ is unspecified)
    
    R[write to console]: Loading required package: S4Vectors
    
    R[write to console]: Loading required package: stats4
    
    R[write to console]: Loading required package: BiocGenerics
    
    R[write to console]: Loading required package: parallel
    
    R[write to console]: 
    Attaching package: ‘BiocGenerics’
    
    
    R[write to console]: The following objects are masked from ‘package:parallel’:
    
        clusterApply, clusterApplyLB, clusterCall, clusterEvalQ,
        clusterExport, clusterMap, parApply, parCapply, parLapply,
        parLapplyLB, parRapply, parSapply, parSapplyLB
    
    
    R[write to console]: The following objects are masked from ‘package:stats’:
    
        IQR, mad, sd, var, xtabs
    
    
    R[write to console]: The following objects are masked from ‘package:base’:
    
        anyDuplicated, append, as.data.frame, basename, cbind, colnames,
        dirname, do.call, duplicated, eval, evalq, Filter, Find, get, grep,
        grepl, intersect, is.unsorted, lapply, Map, mapply, match, mget,
        order, paste, pmax, pmax.int, pmin, pmin.int, Position, rank,
        rbind, Reduce, rownames, sapply, setdiff, sort, table, tapply,
        union, unique, unsplit, which.max, which.min
    
    
    R[write to console]: 
    Attaching package: ‘S4Vectors’
    
    
    R[write to console]: The following object is masked from ‘package:base’:
    
        expand.grid
    
    
    R[write to console]: Loading required package: IRanges
    
    R[write to console]: Loading required package: GenomicRanges
    
    R[write to console]: Loading required package: GenomeInfoDb
    
    R[write to console]: Loading required package: SummarizedExperiment
    
    R[write to console]: Loading required package: MatrixGenerics
    
    R[write to console]: Loading required package: matrixStats
    
    R[write to console]: 
    Attaching package: ‘MatrixGenerics’
    
    
    R[write to console]: The following objects are masked from ‘package:matrixStats’:
    
        colAlls, colAnyNAs, colAnys, colAvgsPerRowSet, colCollapse,
        colCounts, colCummaxs, colCummins, colCumprods, colCumsums,
        colDiffs, colIQRDiffs, colIQRs, colLogSumExps, colMadDiffs,
        colMads, colMaxs, colMeans2, colMedians, colMins, colOrderStats,
        colProds, colQuantiles, colRanges, colRanks, colSdDiffs, colSds,
        colSums2, colTabulates, colVarDiffs, colVars, colWeightedMads,
        colWeightedMeans, colWeightedMedians, colWeightedSds,
        colWeightedVars, rowAlls, rowAnyNAs, rowAnys, rowAvgsPerColSet,
        rowCollapse, rowCounts, rowCummaxs, rowCummins, rowCumprods,
        rowCumsums, rowDiffs, rowIQRDiffs, rowIQRs, rowLogSumExps,
        rowMadDiffs, rowMads, rowMaxs, rowMeans2, rowMedians, rowMins,
        rowOrderStats, rowProds, rowQuantiles, rowRanges, rowRanks,
        rowSdDiffs, rowSds, rowSums2, rowTabulates, rowVarDiffs, rowVars,
        rowWeightedMads, rowWeightedMeans, rowWeightedMedians,
        rowWeightedSds, rowWeightedVars
    
    
    R[write to console]: Loading required package: Biobase
    
    R[write to console]: Welcome to Bioconductor
    
        Vignettes contain introductory material; view with
        'browseVignettes()'. To cite Bioconductor, see
        'citation("Biobase")', and for packages 'citation("pkgname")'.
    
    
    R[write to console]: 
    Attaching package: ‘Biobase’
    
    
    R[write to console]: The following object is masked from ‘package:MatrixGenerics’:
    
        rowMedians
    
    
    R[write to console]: The following objects are masked from ‘package:matrixStats’:
    
        anyMissing, rowMedians
    
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 3 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 1 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 3 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 9 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 9 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 4 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 4 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 3 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 1 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 1 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: -- note: fitType='parametric', but the dispersion trend was not well captured by the
       function: y = a/x + b, and a local regression fit was automatically substituted.
       specify fitType='local' or 'mean' to avoid this message next time.
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: -- note: fitType='parametric', but the dispersion trend was not well captured by the
       function: y = a/x + b, and a local regression fit was automatically substituted.
       specify fitType='local' or 'mean' to avoid this message next time.
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: -- note: fitType='parametric', but the dispersion trend was not well captured by the
       function: y = a/x + b, and a local regression fit was automatically substituted.
       specify fitType='local' or 'mean' to avoid this message next time.
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: -- note: fitType='parametric', but the dispersion trend was not well captured by the
       function: y = a/x + b, and a local regression fit was automatically substituted.
       specify fitType='local' or 'mean' to avoid this message next time.
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: -- note: fitType='parametric', but the dispersion trend was not well captured by the
       function: y = a/x + b, and a local regression fit was automatically substituted.
       specify fitType='local' or 'mean' to avoid this message next time.
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: converting counts to integer mode
    
    R[write to console]:   the design formula contains one or more numeric variables with integer values,
      specifying a model with increasing fold change for higher values.
      did you mean for this to be a factor? if so, first convert
      this variable to a factor using the factor() function
    
    R[write to console]:   the design formula contains one or more numeric variables that have mean or
      standard deviation larger than 5 (an arbitrary threshold to trigger this message).
      it is generally a good idea to center and scale numeric variables in the design
      to improve GLM convergence.
    
    R[write to console]: estimating size factors
    
    R[write to console]: estimating dispersions
    
    R[write to console]: gene-wise dispersion estimates
    
    R[write to console]: mean-dispersion relationship
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    


            Genes Cluster Condition         padj     padjClus    log2FC
    1 XLOC_034855       0       KCl 3.353751e-21 8.719752e-20 -2.636114
    2 XLOC_036454       0       KCl 1.106672e-18 2.877347e-17 -1.654651
    3 XLOC_001437       0       KCl 1.252213e-16 3.255753e-15 -1.739563
    4 XLOC_008048       0       KCl 1.019006e-15 2.649416e-14 -1.785947
    5 XLOC_044068       0       KCl 1.671318e-15 4.345426e-14 -1.664870
    6 XLOC_001436       0       KCl 1.709171e-15 4.443845e-14 -1.697483


##### **Read in and Plot Previously Saved DeSeq2 Results**


```r
%%R
install.packages("UpSetR",repos = "http://cran.us.r-project.org")
```

    R[write to console]: Installing package into ‘/usr/local/lib/R/site-library’
    (as ‘lib’ is unspecified)
    
    R[write to console]: also installing the dependencies ‘gridExtra’, ‘plyr’
    
    
    R[write to console]: trying URL 'http://cran.us.r-project.org/src/contrib/gridExtra_2.3.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 1062844 bytes (1.0 MB)
    
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: 
    
    R[write to console]: downloaded 1.0 MB
    
    
    R[write to console]: trying URL 'http://cran.us.r-project.org/src/contrib/plyr_1.8.6.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 401191 bytes (391 KB)
    
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: 
    
    R[write to console]: downloaded 391 KB
    
    
    R[write to console]: trying URL 'http://cran.us.r-project.org/src/contrib/UpSetR_1.4.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 4194664 bytes (4.0 MB)
    
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: =
    R[write to console]: 
    
    R[write to console]: downloaded 4.0 MB
    
    
    R[write to console]: 
    
    R[write to console]: 
    R[write to console]: The downloaded source packages are in
    	‘/tmp/RtmpA3uFhj/downloaded_packages’
    R[write to console]: 
    R[write to console]: 
    


Create UpSet plot for KCl perturbed cells


```r
%%R -w 20 -h 20 --units in -r 500
#install.packages("UpSetR",repos = "http://cran.us.r-project.org")
#Cite http://people.seas.harvard.edu/~alex/papers/2014_infovis_upset.pdf
library("UpSetR")

deGenesDF <- read.csv(file = 'D1.1818') #'./kallistoDEAnalysis_Stim/deSeq2_deGenesDF_log2FCof1_singleCellReplicates_noShrinkage_subSample.csv'

#Use Bonferronni correction across clusters to filter genes
deGenesDF_toPlot = subset(deGenesDF,padjClus < .05)

kcl_toPlot = subset(deGenesDF_toPlot,Condition =='KCl')
di_toPlot = subset(deGenesDF_toPlot,Condition =='DI')
#deGenesDF_toPlot = subset(deGenesDF_toPlot,abs(log2FC) < 10) #Remove genes from plot that are inflated by zero expression in other cells

# Create empty list to store vectors
vecsToPlot <- list()

#Plot UpSet for KCl-perturbed genes
clusters = unique(kcl_toPlot$Cluster)
for (i in 1:length(clusters)){
    subset = subset(kcl_toPlot,Cluster == clusters[i])
    vecsToPlot[[i]] <- unique(subset$Genes)
}

names(vecsToPlot) <- clusters

upset(fromList(vecsToPlot), sets = as.character(clusters),nintersects= NA,order.by = "freq",
      mainbar.y.label='Number of Genes in Intersection',
      sets.x.label = 'Number of Perturbed Genes',
      text.scale = c(3, 3, 3, 3, 3, 1.3),
      show.numbers = "no",
      point.size = 2.8,
      mb.ratio= c(0.5, 0.5),
      queries = list(list(query = intersects, params = list("18"), color = "firebrick",active = T),
                     list(query = intersects, params = list("19"), color = "firebrick",active = T),
                     list(query = intersects, params = list("22"), color = "firebrick",active = T),
                     list(query = intersects, params = list("23"), color = "firebrick",active = T),
                     list(query = intersects, params = list("17"), color = "firebrick",active = T),
                     list(query = intersects, params = list("14"), color = "firebrick",active = T),
                     list(query = intersects, params = list("12"), color = "firebrick",active = T),
                     list(query = intersects, params = list("9"), color = "firebrick",active = T),
                     list(query = intersects, params = list("6"), color = "firebrick",active = T),
                     list(query = intersects, params = list("15"), color = "firebrick",active = T),
                     list(query = intersects, params = list("10"), color = "firebrick",active = T),
                     list(query = intersects, params = list("13"), color = "firebrick",active = T),
                     list(query = intersects, params = list("16"), color = "firebrick",active = T),
                     list(query = intersects, params = list("11"), color = "firebrick",active = T),
                     list(query = intersects, params = list("7"), color = "firebrick",active = T),
                     list(query = intersects, params = list("4"), color = "firebrick",active = T),
                     list(query = intersects, params = list("3"), color = "firebrick",active = T),
                     list(query = intersects, params = list("5"), color = "firebrick",active = T),
                     list(query = intersects, params = list("25"), color = "firebrick",active = T),
                     list(query = intersects, params = list("0"), color = "firebrick",active = T),
                     list(query = intersects, params = list("2"), color = "firebrick",active = T))) #Add queries

  
```


![png](deSeq2Analysis_StimulationResponse_files/deSeq2Analysis_StimulationResponse_36_0.png)



![png](deSeq2Analysis_StimulationResponse_files/deSeq2Analysis_StimulationResponse_36_1.png)


Create UpSet plot for DI perturbed cells


```r
%%R -w 20 -h 20 --units in -r 500


# Create empty list to store vectors
vecsToPlot <- list()

clusters = unique(di_toPlot$Cluster)
for (i in 1:length(clusters)){
    subset = subset(di_toPlot,Cluster == clusters[i])
    vecsToPlot[[i]] <- unique(subset$Genes)
}

names(vecsToPlot) <- clusters

upset(fromList(vecsToPlot), sets = as.character(clusters),nintersects= NA,order.by = "freq",
      mainbar.y.label='Number of Genes in Intersection',
      sets.x.label = 'Number of Perturbed Genes',
      text.scale = c(3, 3, 3, 3, 3, 1.3),
      show.numbers = "no",
      point.size = 2.8,
      mb.ratio= c(0.5, 0.5),
      queries = list(list(query = intersects, params = list("24"), color = "firebrick",active = T),
                     list(query = intersects, params = list("19"), color = "firebrick",active = T),
                     list(query = intersects, params = list("11"), color = "firebrick",active = T),
                     list(query = intersects, params = list("22"), color = "firebrick",active = T),
                     list(query = intersects, params = list("16"), color = "firebrick",active = T),
                     list(query = intersects, params = list("14"), color = "firebrick",active = T),
                     list(query = intersects, params = list("9"), color = "firebrick",active = T),
                     list(query = intersects, params = list("6"), color = "firebrick",active = T),
                     list(query = intersects, params = list("17"), color = "firebrick",active = T),
                     list(query = intersects, params = list("7"), color = "firebrick",active = T),
                     list(query = intersects, params = list("15"), color = "firebrick",active = T),
                     list(query = intersects, params = list("23"), color = "firebrick",active = T),
                     list(query = intersects, params = list("4"), color = "firebrick",active = T),
                     list(query = intersects, params = list("2"), color = "firebrick",active = T),
                     list(query = intersects, params = list("25"), color = "firebrick",active = T),
                     list(query = intersects, params = list("3"), color = "firebrick",active = T),
                     list(query = intersects, params = list("0"), color = "firebrick",active = T),
                     list(query = intersects, params = list("5"), color = "firebrick",active = T)))
      
      #Add queries
```


![png](deSeq2Analysis_StimulationResponse_files/deSeq2Analysis_StimulationResponse_38_0.png)



```
deseq_df = pd.read_csv('D1.1818',sep=",") #./kallistoDEAnalysis_Stim/deSeq2_deGenesDF_log2FCof1_singleCellReplicates_noShrinkage_subSample.csv
deseq_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>Genes</th>
      <th>Cluster</th>
      <th>Condition</th>
      <th>padj</th>
      <th>padjClus</th>
      <th>log2FC</th>
      <th>orthoGene</th>
      <th>orthoDescr</th>
      <th>pantherID</th>
      <th>pantherDescr</th>
      <th>goTerms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>XLOC_034855</td>
      <td>0</td>
      <td>KCl</td>
      <td>3.353751e-21</td>
      <td>8.719752e-20</td>
      <td>-2.636114</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>XLOC_036454</td>
      <td>0</td>
      <td>KCl</td>
      <td>1.106672e-18</td>
      <td>2.877347e-17</td>
      <td>-1.654651</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>['PTHR10127']</td>
      <td>['DISCOIDIN, CUB, EGF, LAMININ , AND ZINC META...</td>
      <td>['GO:0007498,GO:0005488,GO:0008233,GO:0008237,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>XLOC_001437</td>
      <td>0</td>
      <td>KCl</td>
      <td>1.252213e-16</td>
      <td>3.255753e-15</td>
      <td>-1.739563</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>4</td>
      <td>XLOC_008048</td>
      <td>0</td>
      <td>KCl</td>
      <td>1.019006e-15</td>
      <td>2.649416e-14</td>
      <td>-1.785947</td>
      <td>LOC102723665</td>
      <td>PREDICTED: basic proline-rich protein-like [H...</td>
      <td>['PTHR24637:SF339']</td>
      <td>['CUTICLE COLLAGEN 79-RELATED']</td>
      <td>['GO:0032502,GO:0016337,GO:0007398,GO:0007498,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>XLOC_044068</td>
      <td>0</td>
      <td>KCl</td>
      <td>1.671318e-15</td>
      <td>4.345426e-14</td>
      <td>-1.664870</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[nan]</td>
    </tr>
  </tbody>
</table>
</div>



How gene annotations are added


```
orthoGene = []
orthoDescr = []

pantherNum = []
pantherDescr = []

goTerms = []


for g in deseq_df.Genes:
        
  sub_df = hg_ortho_df[hg_ortho_df.XLOC.isin([g])]
  panth_df = panther_df[panther_df[0].isin([g])]
  go_df = goTerm_df[goTerm_df[0].isin([g])]

  if len(sub_df) > 0:
    #Save first result for gene/description
    orthoGene += [list(sub_df.Gene)[0]]
    orthoDescr += [list(sub_df.Description)[0]]
  else:
    orthoGene += ['NA']
    orthoDescr += ['NA']


  if len(panth_df) > 0:
    #Save first result for gene/description
    pantherNum += [list(panth_df[1])]
    pantherDescr += [list(panth_df[2])]
  else:
    pantherNum += ['NA']
    pantherDescr += ['NA']


  if len(go_df) > 0:
    #Save first result for gene/description
    goTerms += [list(go_df[1])]
  else:
    goTerms += ['NA']
 
deseq_df['orthoGene'] = orthoGene
deseq_df['orthoDescr'] = orthoDescr

deseq_df['pantherID'] = pantherNum
deseq_df['pantherDescr'] = pantherDescr

deseq_df['goTerms'] = goTerms
deseq_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Genes</th>
      <th>Cluster</th>
      <th>Condition</th>
      <th>padj</th>
      <th>padjClus</th>
      <th>log2FC</th>
      <th>orthoGene</th>
      <th>orthoDescr</th>
      <th>pantherID</th>
      <th>pantherDescr</th>
      <th>goTerms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>XLOC_034855</td>
      <td>0</td>
      <td>KCl</td>
      <td>3.353751e-21</td>
      <td>8.719752e-20</td>
      <td>-2.636114</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>XLOC_036454</td>
      <td>0</td>
      <td>KCl</td>
      <td>1.106672e-18</td>
      <td>2.877347e-17</td>
      <td>-1.654651</td>
      <td>NA</td>
      <td>NA</td>
      <td>[PTHR10127]</td>
      <td>[DISCOIDIN, CUB, EGF, LAMININ , AND ZINC METAL...</td>
      <td>[GO:0007498,GO:0005488,GO:0008233,GO:0008237,G...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>XLOC_001437</td>
      <td>0</td>
      <td>KCl</td>
      <td>1.252213e-16</td>
      <td>3.255753e-15</td>
      <td>-1.739563</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>XLOC_008048</td>
      <td>0</td>
      <td>KCl</td>
      <td>1.019006e-15</td>
      <td>2.649416e-14</td>
      <td>-1.785947</td>
      <td>LOC102723665</td>
      <td>PREDICTED: basic proline-rich protein-like [H...</td>
      <td>[PTHR24637:SF339]</td>
      <td>[CUTICLE COLLAGEN 79-RELATED]</td>
      <td>[GO:0032502,GO:0016337,GO:0007398,GO:0007498,G...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>XLOC_044068</td>
      <td>0</td>
      <td>KCl</td>
      <td>1.671318e-15</td>
      <td>4.345426e-14</td>
      <td>-1.664870</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
      <td>[nan]</td>
    </tr>
  </tbody>
</table>
</div>




```
deseq_df.to_csv('stim_deSeq2_deGenesDF_log2FCof1_singleCellReplicates_noShrinkage_subSample_annotations.csv') 
# DOI: D1.1818
```
