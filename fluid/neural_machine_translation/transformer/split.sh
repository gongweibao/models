#!/bin/bash
rm -f ./train.tok.clean.bpe.32000.en-de.train_0
touch ./train.tok.clean.bpe.32000.en-de.train_0

rm -f ./train.tok.clean.bpe.32000.en-de.train_1
touch ./train.tok.clean.bpe.32000.en-de.train_1

line=0
input="./train.tok.clean.bpe.32000.en-de"
while IFS='' read -r var
#while read var
do
  #echo "$var"
  echo $var >> "./train.tok.clean.bpe.32000.en-de.train_"$(($line % 2))
  #echo $line
  line=$((line+1))
done < "$input"
