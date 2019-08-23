#!/bin/bash

#dir=./data/CASPos20190601/

data_group_id=1
root=./data/test_db/group
slash=/
dir=${root}${data_group_id}${slash}
group_dir=group${data_group_id}

for i in `ls ${dir}`
do
    ./db2csv_tool/convert-db-to-csv.sh ${dir}${i}
    file_prefix=$(echo ${i} | cut -d . -f1)
    mv T_Sample.csv ./data/test_csv/group${data_group_id}/${file_prefix}.csv 
done

