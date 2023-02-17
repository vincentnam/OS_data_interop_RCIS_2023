Code repository of : "Interoperability of Open Science Metadata: what about the reality ?", submitted in RCIS 2023 (https://www.rcis-conf.com/rcis2023/callPapers.php).
This repository contains code, data and results of experimentations..

"requirements.txt" is the exact configuration of python packages used. (from "pip freeze > requirements.txt" command)

Step to reproduce :

[matching.py][./matching.py] create results in "results" folder (as a dict (from pickle library) dump). (run the main)
[results/results_analysis.ipynb][./results/results_analysis.ipynb] is used to refactor files and add ground truth associated for each match done. (run each cell)
[formatted_results/formatted_results_analysis.ipynb][./formatted_results/formatted_results_analysis.ipynb] contains all the function to analysis results, get descriptive analysis on it.
[statistical_analysis.R][./statistical_analysis.R] is a R file (see https://www.r-project.org/) to get statistical analysis and correlation on results.
You'll find descriptive CSV in "formatted_results/describe_csv" for each method and couple of schema.

Each couple is called by the first model in couple. ("multi_ODATIS" will described word mover distance on word embedding method for ODATIS / AERIS couple.)

"metric" folder contain word counting in wikipedia for english language and a notebook that show some metrics on models. "models" contains samples used to extract schema of each platform and a notebook to extract each schema. "mappings" folder contain cda2r4 folder with C-CDA transformed files to FHIR, mappings csv and notebook to extract mappings from cda2r4 folder for FHIR/C-CDA couple.

Models are saved in CSV file with 3 columns : unnammed column (path to the metadata value in schema, the concatenation of ,count (count of occurence when extracted from files), values (values possible for each path)

Count and values are not used in scripts but it describes schema (you can set it to empty or 0 values to add new schema). It helps to create mapping.