
<a href="https://colab.research.google.com/github/pachterlab/CWGFLHGCCHAP_2021/blob/master/notebooks/CellAtlasAnalysis/deSeq2Analysis_StarvationResponse.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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
#Starvation h5ad data, all nonzero genes included, filtered for 'real cells' from de-multiplexing
download_file('10.22002/D1.1797','.gz')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=26058.0), HTML(value='')))





    'D1.1797.gz'




```
#CellRanger Starvation h5ad data
download_file('10.22002/D1.1798','.gz')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=45376.0), HTML(value='')))





    'D1.1798.gz'




```
#Kallisto bus clustered starvation data, h5ad
download_file('10.22002/D1.1796','.gz')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=479630.0), HTML(value='')))


    





    'D1.1796.gz'




```
#Saved DeSeq2 Results for Fed/Starved (Differentially expressed under starvation --> perturbed genes)
download_file('10.22002/D1.1810','.gz')
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=224.0), HTML(value='')))





    'D1.1810.gz'




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
!gunzip *.gz
```


```
!pip install --quiet anndata
!pip install --quiet scanpy
!pip3 install --quiet leidenalg
!pip install --quiet louvain
```

    [K     |████████████████████████████████| 122kB 6.7MB/s 
    [K     |████████████████████████████████| 7.7MB 1.4MB/s 
    [K     |████████████████████████████████| 71kB 9.0MB/s 
    [K     |████████████████████████████████| 51kB 7.2MB/s 
    [?25h  Building wheel for sinfo (setup.py) ... [?25l[?25hdone
    [K     |████████████████████████████████| 2.4MB 5.0MB/s 
    [K     |████████████████████████████████| 3.2MB 39.9MB/s 
    [K     |████████████████████████████████| 2.2MB 5.4MB/s 
    [?25h


```
!pip3 install --quiet rpy2
```

### **Import Packages**


```
import pandas as pd
import anndata
import scanpy as sc
import numpy as np
import scipy.sparse

import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import (KNeighborsClassifier,NeighborhoodComponentsAnalysis)
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

import random

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
%matplotlib inline
sc.set_figure_params(dpi=125)

import seaborn as sns
sns.set(style="whitegrid")
%load_ext rpy2.ipython
```


```
# See version of all installed packages, last done 11/27/2020
# !pip list -v > pkg_vers_20201127.txt
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

### **Run DeSeq2 Analysis for Starvation Data**


```
#Remove clusters with < 10 cells per condition

#Read in previously saved data
bus_fs_clus = anndata.read("D1.1796")
print(bus_fs_clus )

bus_fs_raw = anndata.read("D1.1797")

bus_fs_raw = bus_fs_raw[bus_fs_clus.obs_names,]
#bus_fs_raw.obs['orgID'] = bus_fs_clus.obs['orgID']
bus_fs_raw.obs['fed'] = bus_fs_clus.obs['fed']
bus_fs_raw.obs['cellRanger_louvain'] = bus_fs_clus.obs['cellRanger_louvain']
bus_fs_raw


#clusSize
```

    Trying to set attribute `.obs` of view, copying.


    AnnData object with n_obs × n_vars = 13673 × 8696
        obs: 'batch', 'n_counts', 'n_countslog', 'louvain', 'leiden', 'orgID', 'fed', 'starved', 'fed_neighbor_score', 'cellRanger_louvain', 'annos', 'new_cellRanger_louvain', 'annosSub'
        var: 'n_counts', 'mean', 'std'
        uns: 'annosSub_colors', 'annos_colors', 'cellRanger_louvain_colors', 'cellRanger_louvain_sizes', "dendrogram_['new_cellRanger_louvain']", 'dendrogram_new_cellRanger_louvain', 'fed_colors', 'fed_neighbor_score_colors', 'leiden', 'leiden_colors', 'louvain', 'louvain_colors', 'neighbors', 'new_cellRanger_louvain_colors', 'orgID_colors', 'paga', 'pca', 'rank_genes_groups', 'umap'
        obsm: 'X_nca', 'X_pca', 'X_tsne', 'X_umap'
        varm: 'PCs'
        obsp: 'connectivities', 'distances'





    AnnData object with n_obs × n_vars = 13673 × 46716
        obs: 'batch', 'fed', 'cellRanger_louvain'




```
#Determine which clusters large enough to do DE with
def clusToKeep(bus_fs_clus):
  keep = []
  clusSize = {}
  for i in np.unique(bus_fs_clus.obs['cellRanger_louvain']):
      cells = bus_fs_clus[bus_fs_clus.obs['cellRanger_louvain'].isin([i])]
      fed_cells = len(cells[cells.obs['fed']=='True'].obs_names)
      starv_cells = len(cells[cells.obs['fed']=='False'].obs_names)
      min_cells = np.min([fed_cells,starv_cells])
      if min_cells > 10:
        keep += [i]
        #clusSize[i] = min_cells
  return keep
```


```
#Subsample from full dataset, across each cluster
def getSampled_Cluster(bus_fs_clus,bus_fs_raw,keep):

  subSample = 100
  cellNames = np.array(bus_fs_clus.obs_names)
  fed = np.array(list(bus_fs_clus.obs['fed'] == 'True'))
  starv = np.array(list(bus_fs_clus.obs['fed'] == 'False'))
  #di = np.array(list(jelly4Raw.obs['condition'] == 'DI'))

  allCells = []
  for i in keep:
      #subSample =  clusSize[i] # REMOVE IF NOT LOOKING AT SIMILARLY SIZED CLUSTERS
      
      cells = np.array(list(bus_fs_clus.obs['cellRanger_louvain'].isin([i])))
      fed_cells = list(np.where(fed & cells)[0])
      starv_cells = list(np.where(starv & cells)[0])
      
      #Take all cells if < subSample
      if len(fed_cells) >= subSample:
          fed_choice = random.sample(fed_cells,subSample)
      else:
          fed_choice = fed_cells
          
      if len(starv_cells) >= subSample:
          starv_choice = random.sample(starv_cells,subSample)
      else:
          starv_choice = starv_cells
          
          
      pos = list(fed_choice)+list(starv_choice)
      #print(len(pos))
      
      allCells += list(cellNames[pos])

      
  sub_raw = bus_fs_raw[allCells,:]
  return sub_raw
```


```
#For full dataset don't filter by highly variable
keep = clusToKeep(bus_fs_clus)
sub_raw = getSampled_Cluster(bus_fs_clus,bus_fs_raw,keep)
print(sub_raw)

sub_raw_copy = sub_raw.copy()
sc.pp.filter_cells(sub_raw, min_counts=0)
sc.pp.filter_genes(sub_raw, min_counts=0)
sc.pp.normalize_per_cell(sub_raw_copy, counts_per_cell_after=1e4)
sub_raw_copy.raw = sc.pp.log1p(sub_raw_copy, copy=True)

#sc.pp.highly_variable_genes(sub_raw_copy,n_top_genes=5000) #This is just a small example, for full data used all nonzero genes
#sub_raw = sub_raw[:,sub_raw_copy.var['highly_variable']]

```

    Trying to set attribute `.obs` of view, copying.


    View of AnnData object with n_obs × n_vars = 6026 × 46716
        obs: 'batch', 'fed', 'cellRanger_louvain'



```
#Instantiate dataframe with gene names
def makeDF_forR(sub_raw):
  fullDF = pd.DataFrame(scipy.sparse.csr_matrix.toarray(sub_raw.X).T, index = sub_raw.var_names.tolist(), columns= sub_raw.obs_names.tolist())
  conds = sub_raw.obs['fed'].tolist()
  #ids = sub_jelly4Raw.obs['orgID'].tolist()
  clus = sub_raw.obs['cellRanger_louvain'].tolist()

  reps = np.repeat(0,len(sub_raw.obs_names))

  length = len(sub_raw[sub_raw.obs['fed'] == 'True'].obs_names)
  reps[sub_raw.obs['fed'] == 'True'] = range(1,length+1)

  length = len(sub_raw[sub_raw.obs['fed'] == 'False'].obs_names)
  reps[sub_raw.obs['fed'] == 'False'] = range(1,length+1)


  sampleDF = pd.DataFrame({'cell_ID': fullDF.columns}) \
          .assign(condition = conds) \
          .assign(replicate = reps) \
          .assign(cluster = clus) 
  sampleDF.index = sampleDF.cell_ID
  sampleDF.head()

  fullDF.to_csv('fullDF.csv')
  sampleDF.to_csv('sampleDF.csv')
```


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
    1 AGCAGCCTCTGGTATG-1 AGCAGCCTCTGGTATG-1      True         1       0
    2 CGTGAGCGTATATCCG-2 CGTGAGCGTATATCCG-2      True         2       0
    3 CATGGCGTCAGTTGAC-1 CATGGCGTCAGTTGAC-1      True         3       0
    4 CCCAATCGTTGTTTGG-1 CCCAATCGTTGTTTGG-1      True         4       0
    5 ACGAGCCCACATCTTT-2 ACGAGCCCACATCTTT-2      True         5       0
    6 GTCACAAGTCTAGGTT-2 GTCACAAGTCTAGGTT-2      True         6       0



```r
%%R
rownames(sampleDF) <- sampleDF$cell_ID 
#Replace '.' in cell barcodes with '-'
rownames(fullDF) <- fullDF$X
colnames(fullDF) <- gsub("\\.", "-", colnames(fullDF))
fullDF <- subset(fullDF, select = -c(X) )
head(fullDF)

sampleDF <- subset(sampleDF, select = -c(cell_ID.1) )
# head(sampleDF)
sampleDF$condition <- factor(sampleDF$condition, labels = c("starved", "fed"))
```


```r
%%R
#Set up R environment
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
    	‘/tmp/RtmpBlm3aK/downloaded_packages’
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

    Hit:1 http://archive.ubuntu.com/ubuntu bionic InRelease
    Get:2 http://archive.ubuntu.com/ubuntu bionic-updates InRelease [88.7 kB]
    Get:3 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ InRelease [3,626 B]
    Get:4 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic InRelease [15.9 kB]
    Hit:5 http://ppa.launchpad.net/cran/libgit2/ubuntu bionic InRelease
    Get:6 http://archive.ubuntu.com/ubuntu bionic-backports InRelease [74.6 kB]
    Hit:7 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
    Get:8 http://security.ubuntu.com/ubuntu bionic-security InRelease [88.7 kB]
    Ign:9 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
    Ign:10 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
    Hit:11 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release
    Hit:12 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
    Get:13 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ Packages [41.5 kB]
    Get:14 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic/main Sources [1,704 kB]
    Get:15 http://archive.ubuntu.com/ubuntu bionic-updates/universe amd64 Packages [2,140 kB]
    Get:16 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic/main amd64 Packages [873 kB]
    Get:17 http://archive.ubuntu.com/ubuntu bionic-updates/restricted amd64 Packages [277 kB]
    Get:18 http://archive.ubuntu.com/ubuntu bionic-updates/multiverse amd64 Packages [53.4 kB]
    Get:19 http://archive.ubuntu.com/ubuntu bionic-updates/main amd64 Packages [2,273 kB]
    Get:21 http://security.ubuntu.com/ubuntu bionic-security/multiverse amd64 Packages [14.9 kB]
    Get:23 http://security.ubuntu.com/ubuntu bionic-security/restricted amd64 Packages [247 kB]
    Get:24 http://security.ubuntu.com/ubuntu bionic-security/main amd64 Packages [1,846 kB]
    Get:25 http://security.ubuntu.com/ubuntu bionic-security/universe amd64 Packages [1,376 kB]
    Fetched 11.1 MB in 2s (5,148 kB/s)
    Reading package lists... Done
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    libxml2-dev is already the newest version (2.9.4+dfsg1-6.1ubuntu1.3).
    0 upgraded, 0 newly installed, 0 to remove and 26 not upgraded.
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following NEW packages will be installed:
      r-cran-xml
    0 upgraded, 1 newly installed, 0 to remove and 26 not upgraded.
    Need to get 1,711 kB of archives.
    After this operation, 3,072 kB of additional disk space will be used.
    Get:1 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic/main amd64 r-cran-xml amd64 3.99-0.5-1cran1.1804.0 [1,711 kB]
    Fetched 1,711 kB in 0s (15.8 MB/s)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 1.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Selecting previously unselected package r-cran-xml.
    (Reading database ... 145480 files and directories currently installed.)
    Preparing to unpack .../r-cran-xml_3.99-0.5-1cran1.1804.0_amd64.deb ...
    Unpacking r-cran-xml (3.99-0.5-1cran1.1804.0) ...
    Setting up r-cran-xml (3.99-0.5-1cran1.1804.0) ...
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    libcurl4-openssl-dev is already the newest version (7.58.0-2ubuntu3.12).
    0 upgraded, 0 newly installed, 0 to remove and 26 not upgraded.



```r
%%R 
#install.packages("DESeq2",repos = "http://cran.us.r-project.org")
BiocManager::install("DESeq2")
```

    R[write to console]: Bioconductor version 3.12 (BiocManager 1.30.10), R 4.0.3 (2020-10-10)
    
    R[write to console]: Installing package(s) 'BiocVersion', 'DESeq2'
    
    R[write to console]: also installing the dependencies ‘bit’, ‘bitops’, ‘formatR’, ‘bit64’, ‘plogr’, ‘RCurl’, ‘GenomeInfoDbData’, ‘zlibbioc’, ‘matrixStats’, ‘lambda.r’, ‘futile.options’, ‘RSQLite’, ‘xtable’, ‘GenomeInfoDb’, ‘XVector’, ‘MatrixGenerics’, ‘DelayedArray’, ‘futile.logger’, ‘snow’, ‘AnnotationDbi’, ‘annotate’, ‘S4Vectors’, ‘IRanges’, ‘GenomicRanges’, ‘SummarizedExperiment’, ‘BiocGenerics’, ‘Biobase’, ‘BiocParallel’, ‘genefilter’, ‘locfit’, ‘geneplotter’, ‘RcppArmadillo’
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/bit_4.0.4.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 279723 bytes (273 KB)
    
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
    
    R[write to console]: downloaded 273 KB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/bitops_1.0-6.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 8734 bytes
    
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
    
    R[write to console]: downloaded 8734 bytes
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/formatR_1.7.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 105954 bytes (103 KB)
    
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
    
    R[write to console]: downloaded 103 KB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/bit64_4.0.5.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 135091 bytes (131 KB)
    
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
    
    R[write to console]: downloaded 131 KB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/plogr_0.2.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 7795 bytes
    
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
    
    R[write to console]: downloaded 7795 bytes
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/RCurl_1.98-1.2.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 699583 bytes (683 KB)
    
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
    
    R[write to console]: downloaded 683 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/data/annotation/src/contrib/GenomeInfoDbData_1.2.4.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 10673545 bytes (10.2 MB)
    
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
    
    R[write to console]: downloaded 10.2 MB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/zlibbioc_1.36.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 257412 bytes (251 KB)
    
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
    
    R[write to console]: downloaded 251 KB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/matrixStats_0.57.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 188895 bytes (184 KB)
    
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
    
    R[write to console]: downloaded 184 KB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/lambda.r_1.2.4.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 25666 bytes (25 KB)
    
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
    
    R[write to console]: downloaded 25 KB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/futile.options_1.0.1.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 3919 bytes
    
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
    
    R[write to console]: downloaded 3919 bytes
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/RSQLite_2.2.1.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 2429214 bytes (2.3 MB)
    
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
    
    R[write to console]: downloaded 2.3 MB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/xtable_1.8-4.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 564589 bytes (551 KB)
    
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
    
    R[write to console]: downloaded 551 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/GenomeInfoDb_1.26.2.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 3453184 bytes (3.3 MB)
    
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
    
    R[write to console]: downloaded 3.3 MB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/XVector_0.30.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 67789 bytes (66 KB)
    
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
    
    R[write to console]: downloaded 66 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/MatrixGenerics_1.2.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 27612 bytes (26 KB)
    
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
    
    R[write to console]: downloaded 26 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/DelayedArray_0.16.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 579785 bytes (566 KB)
    
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
    
    R[write to console]: downloaded 566 KB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/futile.logger_1.4.3.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 17456 bytes (17 KB)
    
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
    
    R[write to console]: downloaded 17 KB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/snow_0.4-3.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 22675 bytes (22 KB)
    
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
    
    R[write to console]: downloaded 22 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/AnnotationDbi_1.52.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 4338642 bytes (4.1 MB)
    
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
    
    R[write to console]: downloaded 4.1 MB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/annotate_1.68.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 1872445 bytes (1.8 MB)
    
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
    
    R[write to console]: downloaded 1.8 MB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/S4Vectors_0.28.1.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 660079 bytes (644 KB)
    
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
    
    R[write to console]: downloaded 644 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/IRanges_2.24.1.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 452117 bytes (441 KB)
    
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
    
    R[write to console]: downloaded 441 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/GenomicRanges_1.42.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 1175587 bytes (1.1 MB)
    
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
    
    R[write to console]: downloaded 1.1 MB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/SummarizedExperiment_1.20.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 1523133 bytes (1.5 MB)
    
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
    
    R[write to console]: downloaded 1.5 MB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/BiocGenerics_0.36.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 47415 bytes (46 KB)
    
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
    
    R[write to console]: downloaded 46 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/Biobase_2.50.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 1657243 bytes (1.6 MB)
    
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
    
    R[write to console]: downloaded 1.6 MB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/BiocParallel_1.24.1.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 931919 bytes (910 KB)
    
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
    
    R[write to console]: downloaded 910 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/genefilter_1.72.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 1419891 bytes (1.4 MB)
    
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
    
    R[write to console]: downloaded 1.4 MB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/locfit_1.5-9.4.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 200998 bytes (196 KB)
    
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
    
    R[write to console]: downloaded 196 KB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/geneplotter_1.68.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 1433556 bytes (1.4 MB)
    
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
    
    R[write to console]: downloaded 1.4 MB
    
    
    R[write to console]: trying URL 'https://cran.rstudio.com/src/contrib/RcppArmadillo_0.10.1.2.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 1641449 bytes (1.6 MB)
    
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
    
    R[write to console]: downloaded 1.6 MB
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/BiocVersion_3.12.0.tar.gz'
    
    R[write to console]: Content type 'application/x-gzip'
    R[write to console]:  length 981 bytes
    
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
    
    R[write to console]: downloaded 981 bytes
    
    
    R[write to console]: trying URL 'https://bioconductor.org/packages/3.12/bioc/src/contrib/DESeq2_1.30.0.tar.gz'
    
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
    	‘/tmp/RtmpBlm3aK/downloaded_packages’
    R[write to console]: 
    R[write to console]: 
    
    R[write to console]: Old packages: 'ggplot2', 'rlang', 'foreign', 'Matrix'
    



```
#Make output directory
!mkdir kallistoDEAnalysis_Starv
```


```r
%%R
clusters <- unique(sampleDF$cluster)
clusters
```

     [1]  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    [26] 25 26 27 28 29 30 31 32 33 34 35



```r
%%R 
#Run DeSeq2 for each of the cell types (between control and starved cells)
install.packages("DESeq2",repos = "http://cran.us.r-project.org")
library("DESeq2")

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
        dds$condition <- relevel(dds$condition, ref = 'fed')
        dds <- DESeq(dds,test="LRT", reduced=~replicate, sfType="poscounts", useT=TRUE, minmu=1e-6, 
                     minReplicatesForReplace=Inf,betaPrior = FALSE)#parallel = TRUE

        #Starv v Fed results
        res <- results(dds,alpha=0.05,name="condition_starved_vs_fed")
        resLFC <- res 

        resLFC <- na.omit(resLFC)
        resOrdered <- resLFC[resLFC$padj < .05,]
        #Keep log2 fold changes < -1 or > 1
        resOrdered <- resOrdered[abs(resOrdered$log2FoldChange) > 1,] 
        outcomes <- resOrdered[order(resOrdered$padj),]

        Genes <- c(Genes,row.names(outcomes))
        Cluster <- c(Cluster,rep(i,length(row.names(outcomes))))
        Condition <- c(Condition,rep('Starved',length(row.names(outcomes)))) 
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

write.csv(deGenesDF,'./kallistoDEAnalysis_Starv/deSeq2_deGenesDF_log2FCof1_singleCellReplicates_noShrinkage_subSample.csv')

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
    
    R[write to console]: -- note: fitType='parametric', but the dispersion trend was not well captured by the
       function: y = a/x + b, and a local regression fit was automatically substituted.
       specify fitType='local' or 'mean' to avoid this message next time.
    
    R[write to console]: final dispersion estimates
    
    R[write to console]: fitting model and testing
    
    R[write to console]: 2 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
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
    
    R[write to console]: 12 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
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
    
    R[write to console]: 10 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
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
    
    R[write to console]: 8 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
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
    
    R[write to console]: 8 rows did not converge in beta, labelled in mcols(object)$fullBetaConv. Use larger maxit argument with nbinomLRT
    
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
    
    R[write to console]: -- note: fitType='parametric', but the dispersion trend was not well captured by the
       function: y = a/x + b, and a local regression fit was automatically substituted.
       specify fitType='local' or 'mean' to avoid this message next time.
    
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
    1 XLOC_030861       0   Starved 9.125564e-14 3.102692e-12 -1.641450
    2 XLOC_010635       0   Starved 5.121409e-13 1.741279e-11 -1.382850
    3 XLOC_040775       0   Starved 1.881249e-11 6.396248e-10  1.093848
    4 XLOC_012879       0   Starved 9.571692e-11 3.254375e-09 -1.921527
    5 XLOC_028699       0   Starved 1.657337e-10 5.634945e-09 -1.099936
    6 XLOC_011294       0   Starved 1.278944e-09 4.348410e-08 -1.137523



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
    	‘/tmp/Rtmpl94P6z/downloaded_packages’
    R[write to console]: 
    R[write to console]: 
    



```r
%%R -w 20 -h 20 --units in -r 500
#Cite http://people.seas.harvard.edu/~alex/papers/2014_infovis_upset.pdf
library("UpSetR")
deGenesDF <- read.csv(file = './kallistoDEAnalysis_Starv/deSeq2_deGenesDF_log2FCof1_singleCellReplicates_noShrinkage_subSample.csv') #./kallistoDEAnalysis_Starv/deSeq2_deGenesDF_log2FCof1_singleCellReplicates_noShrinkage_subSample.csv

#Bonferronni correction across clusters
deGenesDF_toPlot = subset(deGenesDF,padjClus < .05)

# Create empty list to store vectors
vecsToPlot <- list()

clusters = unique(deGenesDF_toPlot$Cluster)
for (i in 1:length(clusters)){
    subset = subset(deGenesDF_toPlot,Cluster == clusters[i])
    vecsToPlot[[i]] <- unique(subset$Genes)
}

names(vecsToPlot) <- clusters

upset(fromList(vecsToPlot), sets = as.character(clusters),nintersects= NA,order.by = "freq",
      mainbar.y.label='',
      sets.x.label = '',
      text.scale = c(1.5, 2, 1.5, 2, 1.3, 1.3),
      show.numbers = "no",
      point.size = 2.8,
      mb.ratio= c(0.5, 0.5),
      queries = list(list(query = intersects, params = list("9"), color = "firebrick",active = T),
                     list(query = intersects, params = list("28"), color = "firebrick",active = T),
                     list(query = intersects, params = list("21"), color = "firebrick",active = T),
                     list(query = intersects, params = list("17"), color = "firebrick",active = T),
                     list(query = intersects, params = list("6"), color = "firebrick",active = T),
                     list(query = intersects, params = list("31"), color = "firebrick",active = T),
                     list(query = intersects, params = list("24"), color = "firebrick",active = T),
                     list(query = intersects, params = list("8"), color = "firebrick",active = T),
                     list(query = intersects, params = list("15"), color = "firebrick",active = T),
                     list(query = intersects, params = list("10"), color = "firebrick",active = T),
                     list(query = intersects, params = list("23"), color = "firebrick",active = T),
                     list(query = intersects, params = list("26"), color = "firebrick",active = T),
                     list(query = intersects, params = list("11"), color = "firebrick",active = T),
                     list(query = intersects, params = list("33"), color = "firebrick",active = T),
                     list(query = intersects, params = list("12"), color = "firebrick",active = T),
                     list(query = intersects, params = list("19"), color = "firebrick",active = T),
                     list(query = intersects, params = list("29"), color = "firebrick",active = T),
                     list(query = intersects, params = list("14"), color = "firebrick",active = T),
                     list(query = intersects, params = list("30"), color = "firebrick",active = T),
                     list(query = intersects, params = list("18"), color = "firebrick",active = T),
                     list(query = intersects, params = list("3"), color = "firebrick",active = T),
                     list(query = intersects, params = list("22"), color = "firebrick",active = T),
                     list(query = intersects, params = list("27"), color = "firebrick",active = T),
                     list(query = intersects, params = list("0"), color = "firebrick",active = T),
                     list(query = intersects, params = list("34"), color = "firebrick",active = T),
                     list(query = intersects, params = list("32"), color = "firebrick",active = T),
                     list(query = intersects, params = list("16"), color = "firebrick",active = T),
                     list(query = intersects, params = list("35"), color = "firebrick",active = T)))
```


![png](deSeq2Analysis_StarvationResponse_files/deSeq2Analysis_StarvationResponse_31_0.png)



```
deseq_df = pd.read_csv('./kallistoDEAnalysis_Starv/deSeq2_deGenesDF_log2FCof1_singleCellReplicates_noShrinkage_subSample.csv',
            sep=",")
deseq_df.head()
```


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
      <td>XLOC_030861</td>
      <td>0</td>
      <td>Starved</td>
      <td>9.125564e-14</td>
      <td>3.102692e-12</td>
      <td>-1.641450</td>
      <td>SRSF1</td>
      <td>serine/arginine-rich splicing factor 1 isofor...</td>
      <td>[PTHR23147:SF44]</td>
      <td>[SERINE/ARGININE-RICH SPLICING FACTOR 1]</td>
      <td>[nan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>XLOC_010635</td>
      <td>0</td>
      <td>Starved</td>
      <td>5.121409e-13</td>
      <td>1.741279e-11</td>
      <td>-1.382850</td>
      <td>SRSF1</td>
      <td>serine/arginine-rich splicing factor 1 isofor...</td>
      <td>[PTHR23147:SF44]</td>
      <td>[SERINE/ARGININE-RICH SPLICING FACTOR 1]</td>
      <td>[nan]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>XLOC_040775</td>
      <td>0</td>
      <td>Starved</td>
      <td>1.881249e-11</td>
      <td>6.396248e-10</td>
      <td>1.093848</td>
      <td>PINX1</td>
      <td>PIN2/TERF1-interacting telomerase inhibitor 1...</td>
      <td>[PTHR23149:SF27]</td>
      <td>[PIN2/TERF1-INTERACTING TELOMERASE INHIBITOR 1]</td>
      <td>[GO:0030234,GO:0004857,GO:0005515,GO:0003676,G...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>XLOC_012879</td>
      <td>0</td>
      <td>Starved</td>
      <td>9.571692e-11</td>
      <td>3.254375e-09</td>
      <td>-1.921527</td>
      <td>NA</td>
      <td>NA</td>
      <td>[PTHR43056:SF5]</td>
      <td>[ALPHA/BETA-HYDROLASES SUPERFAMILY PROTEIN]</td>
      <td>[GO:0016787,GO:0044238,GO:0019538,GO:0006473,G...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>XLOC_028699</td>
      <td>0</td>
      <td>Starved</td>
      <td>1.657337e-10</td>
      <td>5.634945e-09</td>
      <td>-1.099936</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
  </tbody>
</table>
</div>




```
deseq_df.to_csv('deSeq2_deGenesDF_log2FCof1_singleCellReplicates_noShrinkage_subSample_annotations.csv')
```
