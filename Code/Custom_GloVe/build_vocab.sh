#!/bin/bash

# Note that this script uses GNU-style sed as gsed. On Mac OS, you are required to first https://brew.sh/
#    brew install gnu-sed
# on linux, use sed instead of gsed in the command below:
cat ../clean_pos_train_with_stopWords.tx ../clean_neg_train_with_stopWords.tx | gsed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_full.txt
