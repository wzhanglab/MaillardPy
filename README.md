# Maillard: Building machine learning models for predicting the products of maillard reaction

<p align="center">
  <img src="http://maillard.neau.edu.cn/resimg/generation_pathways_of_ages.png"/>
</p>

## Abstract
The Maillard reaction is a chemical reaction between amino acids and reducing sugars that gives browned food its distinctive flavor. Seared steaks, fried dumplings, cookies and other kinds of biscuits, breads, toasted marshmallows, and many other foods undergo this reaction. It is named after French chemist Louis-Camille Maillard, who first described it in 1912 while attempting to reproduce biological protein synthesis.
Maillard reactions can produce hundreds of different compounds depending on the chemical constituents in the food, the temperature, the time, the presence of air and other factors. 



## Authors
1. Yutang Wang<sup>*</sup>
2. Haibin Ren
3. Huihui Yang

Key Lab of Dairy Science, Ministry of Education, Northeast Agriculture University, education (KLDS-NEAU), Harbin, China
<sup>*</sup>Corresponding Author (wangyt@neau.edu.cn)

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
    ├── data                  # Model Training & Test Data (Tabular Format)
    │   ├── maillard-test.tsv
    │   ├── maillard-train.tsv
    ├── maillard            # All Source Files
    │   ├── models			# Trained Models
    │   │   ├── AGEs_NN.p
    │   │   ├── dic_SVM.p
    │   ├── model.py
    │   ├── read_file.py
    ├── manuscript-experiments           # Testing modules (including those for random-control experiments)
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
Y.Wang designed the study. H.R. curated the data. H.Y., H.R. performed feature selection and importance ranking experiments, and trained the models. H.R. generated the Maillard predictions for specialized chemicals sets.
