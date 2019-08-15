#!/bin/bash

dir=./data/CASPos20190601/
for i in `ls ${dir}`
do
    ./db2csv_tool/convert-db-to-csv.sh ${dir}${i}
    file_prefix=$(echo ${i} | cut -d . -f1)
    mv T_Sample.csv ./data/CASPos20190601_csv/${file_prefix}.csv 
done

