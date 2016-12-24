#!/bin/bash -x

for src in ../input/*.zip; do
  dst=${src/.zip/.gz}
  gunzip -c $src | gzip > $dst
done
