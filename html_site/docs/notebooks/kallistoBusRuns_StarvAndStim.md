
<a href="https://colab.research.google.com/github/pachterlab/CWGFLHGCCHAP_2021/blob/master/notebooks/Preprocessing/cDNAFiltering/kallistoBusRuns_StarvAndStim.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


```
!date
```

### **Download Data**


```
#Install kallisto and bustools

!wget --quiet https://github.com/pachterlab/kallisto/releases/download/v0.46.2/kallisto_linux-v0.46.2.tar.gz
!tar -xf kallisto_linux-v0.46.2.tar.gz
!cp kallisto/kallisto /usr/local/bin/

!wget --quiet https://github.com/BUStools/bustools/releases/download/v0.40.0/bustools_linux-v0.40.0.tar.gz
!tar -xf bustools_linux-v0.40.0.tar.gz
!cp bustools/bustools /usr/local/bin/
```


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
#Get reference data () fastq's)

#Trinity Transcripts
download_file('10.22002/D1.1825','.gz')

#Gff3 (Trinity)
download_file('10.22002/D1.1824','.gz')


```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=23757.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=4972.0), HTML(value='')))





    'D1.1824.gz'




```
# Get doi links for all Starvation cDNA fastq.gz files
starvFiles = []
dois = ['10.22002/D1.1840','10.22002/D1.1841','10.22002/D1.1842','10.22002/D1.1843',
        '10.22002/D1.1844','10.22002/D1.1845','10.22002/D1.1846','10.22002/D1.1847',
        '10.22002/D1.1848','10.22002/D1.1849','10.22002/D1.1850','10.22002/D1.1851',
        '10.22002/D1.1852','10.22002/D1.1853','10.22002/D1.1854','10.22002/D1.1855'] #16 doi numbers
for doi in dois:
  url = 'https://api.datacite.org/dois/'+doi+'/media'
  r = requests.get(url).json()
  netcdf_url = r['data'][0]['attributes']['url']

  starvFiles += [netcdf_url]

s1 = starvFiles[0]
s2 = starvFiles[1]
s3 = starvFiles[2]
s4 = starvFiles[3]
s5 = starvFiles[4]
s6 = starvFiles[5]
s7 = starvFiles[6]
s8 = starvFiles[7]

s9 = starvFiles[8]
s10 = starvFiles[9]
s11 = starvFiles[10]
s12 = starvFiles[11]
s13 = starvFiles[12]
s14 = starvFiles[13]
s15 = starvFiles[14]
s16 = starvFiles[15]
```


```
# Get doi links for all Stimulation cDNA fastq.gz files
stimFiles = []
dois = ['10.22002/D1.1860','10.22002/D1.1863','10.22002/D1.1864','10.22002/D1.1865',
        '10.22002/D1.1866','10.22002/D1.1868','10.22002/D1.1870','10.22002/D1.1871'] #8 numbers
for doi in dois:
  url = 'https://api.datacite.org/dois/'+doi+'/media'
  r = requests.get(url).json()
  netcdf_url = r['data'][0]['attributes']['url']

  stimFiles += [netcdf_url]


stim1 = stimFiles[0]
stim2 = stimFiles[1]
stim3 = stimFiles[2]
stim4 = stimFiles[3]
stim5 = stimFiles[4]
stim6 = stimFiles[5]
stim7 = stimFiles[6]
stim8 = stimFiles[7]

```


```
#Get original, CellRanger clustered starvation adata
download_file('10.22002/D1.1798','.gz')

#Cell barcodes selected from Stim ClickTags
download_file('10.22002/D1.1817','.gz')


```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: TqdmDeprecationWarning: Please use `tqdm.notebook.trange` instead of `tqdm.tnrange`



    HBox(children=(FloatProgress(value=0.0, max=45376.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=183.0), HTML(value='')))





    'D1.1817.gz'




```
!gunzip *.gz
```


```
!pip install --quiet anndata
!pip install --quiet scanpy
!pip install --quiet louvain
```

    [K     |████████████████████████████████| 122kB 7.6MB/s 
    [K     |████████████████████████████████| 7.7MB 6.5MB/s 
    [K     |████████████████████████████████| 51kB 6.6MB/s 
    [K     |████████████████████████████████| 71kB 9.2MB/s 
    [?25h  Building wheel for sinfo (setup.py) ... [?25l[?25hdone
    [K     |████████████████████████████████| 2.2MB 9.6MB/s 
    [K     |████████████████████████████████| 3.2MB 45.6MB/s 
    [?25h

### **Import Packages**


```
import pandas as pd
import anndata
import scanpy as sc
import numpy as np
import scipy.sparse

import matplotlib.pyplot as plt
%matplotlib inline
sc.set_figure_params(dpi=125)
```

### **Run kallisto bus on data with Starvation cDNA data**




```
#Make Kallisto index (referene https://www.kallistobus.tools/getting_started)
!mv D1.1825 transcripts.fa
!kallisto index -i clytia_trin.idx -k 31 transcripts.fa
```

    
    [build] loading fasta file transcripts.fa
    [build] k-mer length: 31
    [build] warning: clipped off poly-A tail (longer than 10)
            from 30 target sequences
    [build] warning: replaced 205989 non-ACGUT characters in the input sequence
            with pseudorandom nucleotides
    [build] counting k-mers ... tcmalloc: large alloc 1610612736 bytes == 0x6a69c000 @  0x7f71df7d61e7 0x6f46bd 0x6f4739 0x4af899 0x4a7a68 0x4aec09 0x44e175 0x7f71de7f2bf7 0x452f19
    done.
    [build] building target de Bruijn graph ...  done 
    [build] creating equivalence classes ...  done
    [build] target de Bruijn graph has 410850 contigs and contains 61068880 k-mers 
    


Run kallisto for one set of samples


```
#Create BUS files from fastq's, can't do separate lines
!mkfifo R1.gz R2.gz R1_02.gz R2_02.gz R1_03.gz R2_03.gz R1_04.gz R2_04.gz; curl -Ls $s1 > R1.gz & curl -Ls $s2 > R2.gz & curl -Ls $s3 > R1_02.gz & curl -Ls $s4 > R2_02.gz & curl -Ls $s5 > R1_03.gz & curl -Ls $s6 > R2_03.gz & curl -Ls $s7 > R1_04.gz & curl -Ls $s8 > R2_04.gz & kallisto bus -i clytia_trin.idx -o bus_output/ -x 10xv2 -t 2 R1.gz R2.gz R1_02.gz R2_02.gz R1_03.gz R2_03.gz R1_04.gz R2_04.gz 

```

    
    [index] k-mer length: 31
    [index] number of targets: 113,915
    [index] number of k-mers: 61,068,880
    tcmalloc: large alloc 1610612736 bytes == 0x2498000 @  0x7fd16fd721e7 0x6f46bd 0x6f4739 0x4af899 0x4a9a10 0x44e2bc 0x7fd16ed8ebf7 0x452f19
    [index] number of equivalence classes: 245,726
    [quant] will process sample 1: R1.gz
                                   R2.gz
    [quant] will process sample 2: R1_02.gz
                                   R2_02.gz
    [quant] will process sample 3: R1_03.gz
                                   R2_03.gz
    [quant] will process sample 4: R1_04.gz
                                   R2_04.gz
    [quant] finding pseudoalignments for the reads ... done
    [quant] processed 219,421,150 reads, 188,296,358 reads pseudoaligned
    



```
#Generate gene-count matrices
!wget --quiet https://github.com/bustools/getting_started/releases/download/getting_started/10xv2_whitelist.txt

#Make t2g file
!mv D1.1824 trinity.gff3
!awk '{ print $12"\t"$10}' trinity.gff3  > t2g_rough.txt
!sed 's/[";]//g' t2g_rough.txt > t2g_trin.txt



```


```
#!cd bus_output/
!mkdir bus_output/genecount/ bus_output/tmp/

!bustools correct -w 10xv2_whitelist.txt -p bus_output/output.bus | bustools sort -T bus_output/tmptmp/ -t 2 -p - | bustools count -o bus_output/genecount/genes -g t2g_trin.txt -e bus_output/matrix.ec -t bus_output/transcripts.txt --genecounts -
```

    Found 737280 barcodes in the whitelist
    Processed 188296358 BUS records
    In whitelist = 179043072
    Corrected = 3009765
    Uncorrected = 6243521
    Read in 182052837 BUS records


Run kallisto for other sample set


```
#Create BUS files from fastq's
!mkfifo R1_new.gz R2_new.gz R1_02_new.gz R2_02_new.gz R1_03_new.gz R2_03_new.gz R1_04_new.gz R2_04_new.gz; curl -Ls $s9 > R1_new.gz & curl -Ls $s10 > R2_new.gz & curl -Ls $s11 > R1_02_new.gz & curl -Ls $s12 > R2_02_new.gz & curl -Ls $s13 > R1_03_new.gz & curl -Ls $s14 > R2_03_new.gz & curl -Ls $s15 > R1_04_new.gz & curl -Ls $s16 > R2_04_new.gz & kallisto bus -i clytia_trin.idx -o bus_output_02/ -x 10xv2 -t 2 R1_new.gz R2_new.gz R1_02_new.gz R2_02_new.gz R1_03_new.gz R2_03_new.gz R1_04_new.gz R2_04_new.gz 
```

    
    [index] k-mer length: 31
    [index] number of targets: 113,915
    [index] number of k-mers: 61,068,880
    tcmalloc: large alloc 1610612736 bytes == 0x31d2000 @  0x7fb6760911e7 0x6f46bd 0x6f4739 0x4af899 0x4a9a10 0x44e2bc 0x7fb6750adbf7 0x452f19
    [index] number of equivalence classes: 245,726
    [quant] will process sample 1: R1_new.gz
                                   R2_new.gz
    [quant] will process sample 2: R1_02_new.gz
                                   R2_02_new.gz
    [quant] will process sample 3: R1_03_new.gz
                                   R2_03_new.gz
    [quant] will process sample 4: R1_04_new.gz
                                   R2_04_new.gz
    [quant] finding pseudoalignments for the reads ... done
    [quant] processed 245,357,482 reads, 211,673,174 reads pseudoaligned
    



```
#Generate gene-count matrices
!cd bus_output_02/
!mkdir bus_output_02/genecount/ bus_output_02/tmp/

!bustools correct -w 10xv2_whitelist.txt -p bus_output_02/output.bus | bustools sort -T bus_output_02/tmp/ -t 2 -p -  | bustools count -o bus_output_02/genecount/genes -g t2g_trin.txt -e bus_output_02/matrix.ec -t bus_output_02/transcripts.txt --genecounts -
```

    Found 737280 barcodes in the whitelist
    Processed 211673174 BUS records
    In whitelist = 201242063
    Corrected = 3408308
    Uncorrected = 7022803
    Read in 204650371 BUS records


Merge matrices (Add -1 to first and -2 to second dataset)


```
path = "bus_output/genecount/"
jelly_adata_01 = sc.read(path+'genes.mtx', cache=True)
jelly_adata_01.var_names = pd.read_csv(path+'genes.genes.txt', header=None)[0]
jelly_adata_01.obs_names = pd.read_csv(path+'genes.barcodes.txt', header=None)[0]

jelly_adata_01.obs_names = [i+"-1" for i in jelly_adata_01.obs_names]
```


```
path = "bus_output_02/genecount/"
jelly_adata_02 = sc.read(path+'genes.mtx', cache=True)
jelly_adata_02.var_names = pd.read_csv(path+'genes.genes.txt', header=None)[0]
jelly_adata_02.obs_names = pd.read_csv(path+'genes.barcodes.txt', header=None)[0]

jelly_adata_02.obs_names = [i+"-2" for i in jelly_adata_02.obs_names]
```


```
jelly_adata = jelly_adata_01.concatenate(jelly_adata_02,join='outer', index_unique=None)
```

Filter cell barcodes by previous selection done with ClickTag counts


```
# Filter barcodes by 'real' cells
cellR = anndata.read('D1.1798')
cells = list(cellR.obs_names)

jelly_adata = jelly_adata[cells,:]
jelly_adata
```




    View of AnnData object with n_obs × n_vars = 13673 × 46716
        obs: 'batch'




```
jelly_adata.write('fedStarved_raw.h5ad')
```

### **Run kallisto bus on data with Stimulation cDNA data**




Run kallisto for one set of samples


```
#Create BUS files from fastq's, can't do separate lines
!mkfifo R1_stim.gz R2_stim.gz R1_02_stim.gz R2_02_stim.gz ; curl -Ls $stim1 > R1_stim.gz & curl -Ls $stim2 > R2_stim.gz & curl -Ls $stim3 > R1_02_stim.gz & curl -Ls $stim4 > R2_02_stim.gz & kallisto bus -i clytia_trin.idx -o bus_output/ -x 10xv3 -t 2 R1_stim.gz R2_stim.gz R1_02_stim.gz R2_02_stim.gz


```


```
#Generate gene-count matrices
!wget --quiet https://github.com/bustools/getting_started/releases/download/getting_started/10xv3_whitelist.txt

```


```
#!cd bus_output/
!mkdir bus_output/genecount/ bus_output/tmp/

!bustools correct -w 10xv3_whitelist.txt -p bus_output/output.bus | bustools sort -T bus_output/tmptmp/ -t 2 -p - | bustools count -o bus_output/genecount/genes -g t2g_trin.txt -e bus_output/matrix.ec -t bus_output/transcripts.txt --genecounts -
```


```
!ls test
```

Run kallisto for other sample set


```
#Create BUS files from fastq's

!mkfifo R1_stim2.gz R2_stim2.gz R1_02_stim2.gz R2_02_stim2.gz ; curl -Ls $stim5 > R1_stim2.gz & curl -Ls $stim6 > R2_stim2.gz & curl -Ls $stim7 > R1_02_stim2.gz & curl -Ls $stim8 > R2_02_stim2.gz & kallisto bus -i clytia_trin.idx -o bus_output_02/ -x 10xv3 -t 2 R1_stim2.gz R2_stim2.gz R1_02_stim2.gz R2_02_stim2.gz

```


```
#Generate gene-count matrices
!cd bus_output_02/
!mkdir bus_output_02/genecount/ bus_output_02/tmp/

!bustools correct -w 10xv3_whitelist.txt -p bus_output_02/output.bus | bustools sort -T bus_output_02/tmp/ -t 2 -p -  | bustools count -o bus_output_02/genecount/genes -g t2g_trin.txt -e bus_output_02/matrix.ec -t bus_output_02/transcripts.txt --genecounts -
```

Merge matrices (Add -1 to first and -2 to second dataset)


```
path = "bus_output/genecount/"
jelly_adata_01 = sc.read(path+'genes.mtx', cache=True)
jelly_adata_01.var_names = pd.read_csv(path+'genes.genes.txt', header=None)[0]
jelly_adata_01.obs_names = pd.read_csv(path+'genes.barcodes.txt', header=None)[0]

jelly_adata_01.obs_names = [i+"-1" for i in jelly_adata_01.obs_names]
```


```
path = "bus_output_02/genecount/"
jelly_adata_02 = sc.read(path+'genes.mtx', cache=True)
jelly_adata_02.var_names = pd.read_csv(path+'genes.genes.txt', header=None)[0]
jelly_adata_02.obs_names = pd.read_csv(path+'genes.barcodes.txt', header=None)[0]

jelly_adata_02.obs_names = [i+"-2" for i in jelly_adata_02.obs_names]
```


```
jelly_adata = jelly_adata_01.concatenate(jelly_adata_02,join='outer', index_unique=None)
```

Filter cell barcodes by previous filtering done with ClickTag counts


```
# Filter barcodes by 'real' cells
!mv D1.1817 jelly4stim_individs_tagCells_50k.mat
barcodes_list = sio.loadmat('jelly4stim_individs_tagCells_50k.mat')
barcodes_list.pop('__header__', None)
barcodes_list.pop('__version__', None)
barcodes_list.pop('__globals__', None)

# Add all cell barcodes for each individual
barcodes = []
for b in barcodes_list:
    if barcodes_list[b] != "None":
        barcodes.append(b)

print(len(barcodes))

barcodes = [s.replace('-1', '-3') for s in barcodes]
barcodes = [s.replace('-2', '-1') for s in barcodes]
barcodes = [s.replace('-3', '-2') for s in barcodes]

```


```
jelly_adata = jelly_adata[barcodes,:]
```


```
jelly_adata.write('stimulation_raw.h5ad')
```
