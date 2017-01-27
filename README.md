# Outbrain Click Prediction 2nd Place Solution (Single model)


## Data location

This code suppose what zipped competition data is located in `../input` directory.


## Requirements

This repository contains code in C++, Python and R. It requires:

* Python - `pandas`, `sklearn`, `numba`
* C++ - `boost`, `boost-iostreams`, `boost-program-options`
* R - `data.table`, will be installed automatically

We tested it in AWS `r4.xlarge` with Ubuntu 16.04 image, and following commands are enough to install all requirements:

    sudo apt-get install python-pip llvm libboost-dev libboost-iostreams-dev libboost-program-options-dev g++ r-base

    sudo pip install enum34
    sudo pip install pandas sklearn numba


## Execution

After input data is located in `../input` and all requirements are installed, full pipeline with validation and submission file generation may be executed by calling `run.sh` script in current directory:

    ./run.sh

The submission file will be placed to `subm` subdir, raw predictions - to `preds` subdir.


## File structure


    +- bin  -  Compiled binaries directory
    +- cache  -  Intermediate feature files directory
    +- logs  -  Text logs directory
    +- preds  -  Raw predictions directory
    +- subm  -  Submission files directory
    |
    +- util  -  Various helper code - functions to read data, evaluate local predictions and so on
    |  |
    |  +- __init__.py  -  Python helper functions (score computation, generic model training pipeline)
    |  +- meta.py  -  Metainformation about dataset itself (input directory location, split file names, train/test split date)
    |  |
    |  +- io.h  -  C++ functions for reading compressed csv files line-by-line and into various structures
    |  +- data.h  -  C++ functions for reading indexed feature files and corresponding data structures
    |  +- generation.h  -  Generic model data generation pipeline in C++
    |  +- helpers.h  -  Various smaller C++ helper functions, related to model data generation
    |  +- model-helpers.h  -  Smaller C++ helper functions, related to the FFM implementation itself
    |
    +- README.md  -  This file
    +- Makefile  -  Build script for C++ files
    +- run.sh  -  Main script to execute full model pipeline at once
    |
    +- ffm-io.cpp  -  C++ code for reading/writing in binary format used by custom FFM implementation
    +- ffm-model.cpp  -  Custom FFM model implementation, core code which implements score prediction for a given example and updating weights
    +- ffm-model.h  - Header for the FFM model implementation
    +- ffm.cpp  -  Custom FFM model executable code - loading data, iterating over it in batches, saving predictions
    +- ffm.h  -  Header for FFM model executable
    |
    +- prepare-split.py  -  Prepare validation split files
    +- prepare-documents.py  -  Reformat and enrich documents meta, convert timestamps
    +- prepare-events.py  -  Reformat and enrich events, split location into fields, convert uuid to numeric
    +- prepare-counts.py  -  Calculate how ofthen different objects (ads, users, documents) appeared in train+test
    +- prepare-doc-ad-others.R  -  Compute interactions of display_id and ads in display
    +- prepare-viewed-doc-ids.py  -  Compute document ids, viewed by each user
    +- prepare-viewed-doc-sources.py  -  Compute document id sources, viewed by each user
    +- prepare-viewed-docs-one-hour.py  -  Computed ids od documents viewed by user one hour after ad display
    |
    +- prepare-leak.cpp  -  Compute "leak" feature - did user viewed document representing current ad
    +- prepare-rivals.cpp  -  Compute sets of other ads in the same display
    +- prepare-viewed-ads.cpp  -  Compute number of times user viewed (in past and future) specific ads, documents, document sources and so on in the ads
    +- prepare-viewed-docs.cpp  -  Compute number of times user viewed (in past and future) specific documents, document categories, topics and so on
