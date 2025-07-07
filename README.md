# PCID
Steps for conditionally generation:

1. create dictionaries 'DataA_InputData/', 'DataB_ProcessedData/', 'DataC_SavedModel/', 'DataD_OutputData/', 'ZDataD_Molecules', 'ZdataE_AnalysisResult' and 'ZDataU_Param' in the root dictionary.
2. generate dataset for the regress model by running A_Preprocess/AA_read_data_seperate_ion.py.
3. (Optional) optimize the hyperparameters for regress model by setting hyperparameters in line 19-21 and running B_Train/BA_regress/BAB_optim_hyperparam.py
4. train the regress model by running B_Train/BA_regress/BAA_regress_separate_ion_V2.py. Notice: in certain task, other regress model may perform better than the model used in this work.
5. fill the missing properties data by running B_Train/BA_regress/BAC_unsparse_data_separate.py.
6. train the embedding model by running B_Train/BB_encode/BBA_train_encoding.py.
7. generate dataset for the diffusion model by running B_Train/BB_encode/BBB_save_encoding.py.
8. train the diffusion model by running B_Train/BC_diffusion/BCAA_train_separate.py.
9. generate ILs by setting generate condition (labels) in line 43 and running C_Evaluation/CCZ_generate.py. The result will be shown in 'ZDataD_Molecules'.
