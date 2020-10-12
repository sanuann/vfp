# Vocal Fold Paralysis (VFP)

Code for Low et al. Uncovering the important acoustic features for detecting vocal fold paralysis with explainable machine learning. 

Note: "speech" refers to "reading" task in manuscript.

## 1. Data

(TODO) Available at Open Science Framework: 

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


We removed 24 patients that were recorded using a different device (an iPad). If performance drops significantly, then the original dataset may be using recording set up to dissociate groups (i.e., if features related to iPad are within certain range determined by iPad, then prediction equals patient).
Patients recorded with iPad are: `[3,4,5,8,9,12,13,18,24,27,28,29,31,33,38,53,54,55,56,64,65,66,71,74]`  
```
egemaps_vector_both_wo-24-patients.csv
egemaps_vector_speech_wo-24-patients.csv
egemaps_vector_vowel_wo-24-patients.csv
```

## 2. Code 
 

(TODO: Add requirements pydra+edaic)

**Figure 1:**
* `./mfcc.ipynb/` 

* `collinearity.py` remove redudant features (reduce multicollinearity)

**Figure 4 and data used for other codes:**
* `./vfp_v7_indfact/` 


    * `run_collinearity_job_array_{data_type}_if.sh` (for `data_type` in ['speech', 'vowel', 'both']) slurm job to run [pydra-ml](https://github.com/nipype/pydra-ml) spec files, one `spec_id` in `range(1,10)` corresponding to dcorrs thresholds of `np.arange(0.2, 1.1, 0.1)`containing: 
    ```
    $ pydraml -s specs/vfp_spec_4models_both_if_{spec_id}.json
    ```

**Figure 2 and Sup. Fig. 1-8**
* `redudant_features.ipynb` Clustermap

**Figure 3 and Tables 2, 3, 4:**
* `analyze_results.py` takes `out-*` files from pydra-ml and produces summaries which were then concatenated into tables. 



**Figure 5**
* `shap_analysis.ipynb` 









