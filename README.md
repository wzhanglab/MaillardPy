# Maillard: Building machine learning models for predicting the products of maillard reaction

<p align="center">
  <img src="http://maillard.neau.edu.cn/resimg/generation_pathways_of_ages.png"/>
</p>

## Abstract
The Maillard reaction is a chemical reaction between amino acids and reducing sugars that gives browned food its distinctive flavor. Seared steaks, fried dumplings, cookies and other kinds of biscuits, breads, toasted marshmallows, and many other foods undergo this reaction. It is named after French chemist Louis-Camille Maillard, who first described it in 1912 while attempting to reproduce biological protein synthesis.
The reactive carbonyl group of the sugar reacts with the nucleophilic amino group of the amino acid,  and forms a complex mixture of poorly characterized molecules responsible for a range of aromas and flavors. 
Maillard reactions can produce hundreds of different Advanced glycation end products (AGEs depending on the chemical constituents in the food, the temperature, the cooking time, and the presence of air. Advanced glycation end products (AGEs) are proteins or lipids that become glycated as a result of exposure to sugars. They are a bio-marker implicated in aging and the development, or worsening, of many degenerative diseases, such as diabetes, atherosclerosis, chronic kidney disease, and Alzheimer's disease.
Melanoidins are brown, high molecular weight heterogeneous polymers that are formed when sugars and amino acids combine (through the Maillard reaction) at high temperatures and low water activity. Melanoidins are commonly present in foods that have undergone some form of non-enzymatic browning, such as barley malts (Vienna and Munich), bread crust, bakery products and coffee. They are also present in the wastewater of sugar refineries, necessitating treatment in order to avoid contamination around the outflow of these refineries.
The polymers make the constituting dietary sugars and fats unavailable to the normal carbohydrate and fat metabolism. Dietary melanoidins themselves produce various effects in the organism: they decrease Phase I liver enzyme activity and promote glycation in vivo, which may contribute to diabetes, reduced vascular compliance and Alzheimer's disease. 



## Authors
1. Yutang Wang<sup>*</sup>
2. Haibin Ren
3. Huihui Yang

Key Lab of Dairy Science, Ministry of Education, Northeast Agriculture University, education (KLDS-NEAU), Harbin, China
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
    ├── data                  # Model Training & Test Data (Tabular Format)
    │   ├── maillard-test.tsv
    │   ├── maillard-train.tsv
    ├── maillard            # All Source Files
    │   ├── models			# Trained Models
    │   │   ├── AGEs_NN.p
    │   │   ├── dic_SVM.p
    │   ├── __init__.py
    │   ├── model.py
    │   ├── properties.py
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
