#!/bin/bash

# Set input directory path
export INPUT=../input

# Compile binaries
make

# Repack all input data to gz (as it simpler to work with from C++)
for src in ../input/*.zip; do
  dst=${src/.zip/.gz}
  gunzip -c $src | gzip > $dst
done

# Prepare indexed feature files
python prepare-events.py
python prepare-documents.py

Rscript feat_disp_ad_doc_others.R
python feat_Uuid_Source_id.py
python feat_Uuid_Doc_Srce_id.py
python feat_Uuid_OneHour_Range.py

# Prepare positional feature files
python prepare-split.py
bin/prepare-leak
bin/prepare-rivals
bin/prepare-similarity
bin/prepare-viewed-ads
bin/prepare-viewed-docs

# Compute counts
bin/prepare-counts

# Train a model and generate submission
python train-model.py
