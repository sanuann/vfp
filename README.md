# Detecting Vocal Fold Paralysis (VFP) with machine learning

**Cite this paper if using the data:**

Low, D. M., Randolph, G., Rao, V., Ghosh, S. S.* & Song, P., C.* (2020). Uncovering the important acoustic features for detecting vocal fold paralysis with explainable machine learning. MedRxiv. 



Note: 
* "speech" refers to "reading" task in manuscript.
* The original audio wav files cannot be shared due to consent restrictions. Here we provide the extracted eGeMAPS features (see manuscript for details).

## 1. Data

Available at [Open Science Framework](https://osf.io/7q2ux/)

#### 1.1. Demographic information 
* `./data/input/VFP_DeidentifiedDemographics.csv` De-identified demographic information
 
#### 1.2. eGeMAPS features, ids, and labels 
**Columns:**
* `sid` subject ID. Important for group shuffle split.
* `filename` from which wav file were features extracted
* `token` what type of sample (speechN, vowelN where N is sample number)
* `target` label to be predicted

```
egemaps_vector_both.csv
egemaps_vector_speech.csv
egemaps_vector_vowel.csv
```




## 2. Code and instructions for reproducibility 
 
To run the `.py` (inluding pydra-ml package) or the `.ipynb` on Jupter Notebook, create a virtual environment and install the `requirements.txt`:
* `conda create --name pydra_vfp --file requirements.txt`
* `conda activate pydra_vfp`


#### 2.1. Figure 1:
* `./mfcc.ipynb/`
* `./data/input/rainbow.wav` audio file used
* `./data/input/rainbow_f0.txt` f0 over time from PRAAT 

#### 2.2. Table 1: Sample sizes and demographic information. 
* `./data/input/VFP_DeidentifiedDemographics.csv` De-identified demographic information
* `demographics.py` Script to obtain info for Table 1.

#### 2.3. Figure 2 and Sup. Fig. 1-8
* `collinearity.py` remove redudant features (reduce multicollinearity) 
* `redudant_features.ipynb` Clustermap

#### 2.4. Run models: output Figure 4 and data used for other codes:
We ran models using [pydra-ml](https://github.com/nipype/pydra-ml) for which a spec file is needed where the dataset is specified. The dataset needs to be in the same dir where the spec file is run. Since we ran models on a cluster, we have SLURM scripts, so  the dataset is in the same dir as the SLURM scripts.
`if` and `indfact` stand for independence factor, the algorithm we created for removing redundant features. 
* `./vfp_v7_indfact/` dir 
    * `/specs/` spec files
    * `run_collinearity_job_array_{data_type}_if.sh` SLURM script to run pydra-ml spec files where `data_type` is 'speech', 'vowel' or 'both'. 
        * ```$ pydraml -s specs/vfp_spec_4models_both_if_{spec_id}.json``` where `spec_id` is value in `range(1,10)` corresponding to dcorrs thresholds of `np.arange(0.2, 1.1, 0.1)` (i.e., we removed redudant features according to the dcor threshold). The job array runs the different spec files in parallel.
    * `thresholds_if_Nvars_{data_type}.txt` were used to build those spec files.
    * `run_clear_locks.sh` runs `clear_locks.py` Run this this if you want to re-run model with different specs (pydra-ml will re-use cache-wf)
    * `run_collinearity_speech_explanations.sh` re-runs models setting `gen_shap` to true in spec files to output SHAP values/explanations.
    * `./outputs/` each run will output a dir such as `out-vfp_spec_4models_both_if_1.json-20200910T024552.823868` with the name of the spec file. 
    
* `./vfp_v8_top5/` runs top 5 featurs specificed in spec files
   

#### 2.5. Figure 3 and Table 2:
* `analyze_results.py` takes `outputs/out-*` files from pydra-ml and produces summaries which were then concatenated into table 2. 

#### 2.6. Figure 5
* `shap_analysis.ipynb` 

#### 2.7. Figure 6
* `./vfp_v8_top1outof5/` runs one of the top 5 features at a time.
* `shap_analysis.ipynb` makes the plots

#### 2.8. Supplementary Table S1
* `collinearity.py` Remove redudant features through nested crossvalidation approach

#### 2.9. Supplementary Table S2
We removed 24 patients that were recorded using a different device (an iPad). If performance drops significantly, then the original dataset may be using recording set up to dissociate groups (i.e., if features related to iPad are within certain range determined by iPad, then prediction equals patient).
Patients recorded with iPad are: `[3,4,5,8,9,12,13,18,24,27,28,29,31,33,38,53,54,55,56,64,65,66,71,74]`  
* `./data/input/features/egemaps_vector_both_wo-24-patients.csv` dataset
* `./data/output/vfp_v8_wo-24-patients/` pydra-ml scripts
* ```
    egemaps_vector_both_wo-24-patients.csv
    egemaps_vector_speech_wo-24-patients.csv
    egemaps_vector_vowel_wo-24-patients.csv
    ```




