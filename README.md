# Analysis of Multiplexed, Single-Cell *Clytia* Medusae Experiments
Notebooks for reproducing all figures and analysis in the *Whole Animal Multiplexed Single-Cell RNA-Seq Reveals Plasticity of *Clytia* Medusae Cell Types* preprint.

## Getting Started

All notebooks, saved as .ipynb's, can be run from Google Colab. Colab links are included in every notebook.

All saved/processed data used for analysis is streamed to the notebooks from [CaltechData](https://data.caltech.edu/).

## Notebooks Directory Contents

1) Preprocessing - All notebooks for ClickTag and cDNA Library Demultiplexing

    a) ClickTagDemultiplexing
    
      * cellRangerClickTagCounts.ipynb
      
      * kallistobusStarvClickTagsProcessing.ipynb
      
      * kallistobusStimClickTags.ipynb
    
    b) cDNAFiltering
    
      * filterStarvCells_ClickTags.ipynb
      
      * kallistoBusRuns_StarvAndStim.ipynb
	
2) CellAtlasAnalysis  - All notebooks for Clustering and Perturbation Response Analysis for Starvation Experiment

    * cellRangerClustering_Starvation.ipynb
    
    * starvation_Analysis.ipynb
    
    * deSeq2Analysis_StarvationResponse.ipynb
    
    * neuronSubpop_Analysis.ipynb
    
    * pseudotime_Analysis.ipynb
    
    
3) ComparativeDistanceAnalysis - All Distance-based Analysis for Cell Type/State Delineation and Cross-Experiment Batch Effects
  
    * allDistanceCalculations.ipynb
 
4) StimulationAnalysis - Clustering and Perturbation Response Analysis for Stimulation Experiment

    * stimulation_Analysis.ipynb
    
    * deSeq2Analysis_StimulationResponse.ipynb
   
   
*---------- For (User) Gene Searching and Plotting ----------*

5) SearchAndPlotInteractive - Notebooks for Exploring and Searching Cell Atlas

    * MARIMBAAnnosAnalysis.ipynb
    
    * exploreMARIMBAData.ipynb
    
    * cellAtlasSearchAndPlot.ipynb




## Authors

* Tara Chari

## Acknowledgments

Several of the pre-processing workflows and initial analyses for the project were first implemented by Jase Gehring.







