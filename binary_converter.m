function sample_gen_from_set

clear;clc; clearvars; close all; fclose all;

root='data/5-fold-mat/'

for i=1:5
    folder_name=strcat(num2str(i),'-fold/')
    file_path=strcat(root,folder_name,'/')
    X_train_pos=load(strcat(file_path,'data.mat'),'X_train_pos')
    X_train_neg=load(strcat(file_path,'data.mat'),'X_train_neg')
    X_test=load(strcat(file_path,'data.mat'),'X_test')
    
    export_X_train_pos_file_path=strcat(file_path,'X_train_pos.dbin')
    export_X_train_neg_file_path=strcat(file_path,'X_train_neg.dbin')
    export_X_test_file_path=strcat(file_path,'X_test_neg.dbin')

    % X_train_pos
    X_train_pos_fid = fopen(export_X_train_pos_file_path, 'wb') ;
    for sample_index = 1:size(X_train_pos.X_train_pos,1)
        for row_index = 1:size(X_train_pos.X_train_pos,2) % 10
            vec_rssi = X_train_pos.X_train_pos(sample_index,row_index,:) % vec_rssi length: 400
            sample_tmp(row_index, :) = int8( vec_rssi );
        end
        label = 0 ; qty = 1 ; sample2write = reshape(sample_tmp', 1, [] ) ; 
        fwrite(X_train_pos_fid, [label, qty, sample2write], 'int8') ;
    end
    fclose(X_train_pos_fid);
    
    % X_train_neg
    X_train_neg_fid = fopen(export_X_train_neg_file_path, 'wb') ;
    for sample_index = 1:size(X_train_neg.X_train_neg,1)
        for row_index = 1:size(X_train_neg.X_train_neg,2) % 10
            vec_rssi = X_train_neg.X_train_neg(sample_index,row_index,:) % vec_rssi length: 400
            sample_tmp(row_index, :) = int8( vec_rssi );
        end
        label = 0 ; qty = 1 ; sample2write = reshape(sample_tmp', 1, [] ) ;
        fwrite(X_train_neg_fid, [label, qty, sample2write], 'int8') ;
    end
    fclose(X_train_neg_fid);
    
    % X_test
    X_test_fid = fopen(export_X_test_file_path, 'wb') ;
    for sample_index = 1:size(X_test.X_test,1)
        for row_index = 1:size(X_test.X_test,2) % 10
            vec_rssi = X_test.X_test(sample_index,row_index,:) % vec_rssi length: 400
            sample_tmp(row_index, :) = int8( vec_rssi );
        end
        label = 0 ; qty = 1 ; sample2write = reshape(sample_tmp', 1, [] ) ;
        fwrite(X_test_fid, [label, qty, sample2write], 'int8') ;
    end
    fclose(X_test_fid);
end
