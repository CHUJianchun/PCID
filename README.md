# PCID
Steps for conditionally generation:

1. create dictionaries 'DataA_InputData/', 'DataB_ProcessedData/', 'DataC_SavedModel/', 'DataD_OutputData/', 'ZDataD_Molecules', 'ZdataE_AnalysisResult' and 'ZDataU_Param' in the root dictionary.
2. download ZINC dataset from https://zinc15.docking.org/ and copy the files to 'DataA_InputData/'.
3. Train the submodules step by step.
4. generate.
