# Maillard: Building machine learning models for predicting the products of maillard reaction

<p align="center">
  <img src="https://maillard.neau.edu.cn/"/>
</p>

## Abstract
The dichotomy of sweet and bitter tastes is a salient evolutionary feature of human
gustatory system with an innate attraction to sweet taste and aversion to bitterness. A better
understanding of molecular correlates of bitter-sweet taste gradient is crucial for identification of
natural as well as synthetic compounds of desirable taste on this axis. While previous studies have
advanced our understanding of the molecular basis of bitter-sweet taste and contributed models for
their identification, there is ample scope to enhance these models by meticulous compilation of bitter-
sweet molecules and utilization of a wide spectrum of molecular descriptors. Towards these goals,
based on structured data compilation our study provides an integrative framework with state-of-the-art
machine learning models for bitter-sweet taste prediction (BitterSweet). We compare different sets of
molecular descriptors for their predictive performance and further identify important features as well
as feature blocks. The utility of BitterSweet models is demonstrated by taste prediction on large
specialized chemical sets such as FlavorDB, FooDB, SuperSweet, Super Natural II, DSSTox, and
DrugBank. To facilitate future research in this direction, we make all datasets and BitterSweet models
publicly available, and also present an end-to-end software for bitter-sweet taste prediction based on
freely available chemical descriptors.

## Authors
1. Yutang Wang
2. Haibin Ren
3. Huihui Yang<sup>*</sup>

Key Lab of Dairy Science (KLDS-NEAU), Harbin, China
<sup>*</sup>Corresponding Author (fic@neau.edu.cn, wangyt@neau.edu.cn)

## Pre-requisites for execution
To setup a working environment to execute some or all sections of this project, you must:

1. Clone the project `MaillardPy` - 
	    
	    $ git clone https://github.com/wzhanglab/MaillardPy
	    $ cd maillard

2. We use `conda` as a tool to create isolated virtual environments and since some of our packages require building binaries from their source, it is necessary to create your env from the `requirement.yml` file provided.

	 	$ conda env create -f environment.yml
	 	$ conda activate env 
	 	
3. To deactivate this environment after usage - 
		
		$ conda deactivate
		
\* Ensure that all scripts are run under a python 3.7 environment.

## Directory Structure

    .
    .
    ├── data                     # Model Training & Test Data (Tabular Format)
    │   ├── maillard-test.tsv
    │   ├── maillard-train.tsv
    ├── bittersweet                    # All Source Files
    │   ├── models			# Trained Models
    │   │   ├── maillard_chemopy_boruta_features.p
    │   │   ├── maillard_chemopy_rf_boruta.p
    │   ├── __init__.py
    │   ├── model.py
    │   ├── properties.py
    │   ├── read_file.py
    ├── manuscript-experiments                    # Testing modules (including those for random-control experiments)
    │   ├── maillard					# Directory containing scripts
    │   ├── data						# Directory containing data
    │   ├── models						# Directory containing models
    ├── examples             
    ├── predict.py 						# methods to test our models
    .
    .


## Acknowledgement
The authors thank Center for Education Technology,NEAU(CET-NEAU) for providing computational facilities and support. 

## Author Contributions
Y.Wang. and H.Y. designed the study. H.R. curated the data. H.Y., H.R. performed feature selection and importance ranking experiments, and trained the models. H.R. generated the Maillard predictions for specialized chemicals sets. All the authors analysed the results and wrote the manuscript.  
