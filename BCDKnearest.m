ds_train = importdata('wdbc_train.data');
dp_input_train = ds_train(:,2:end);
dp_output_train = ds_train(:,1);

ds_test = importdata('wdbc_test.data');
dp_input_test = ds_test(:,2:end);
dp_output_test = ds_test(:,1);

ds_valid = importdata('wdbc_valid.data');
dp_input_valid = ds_valid(:,2:end);
dp_output_valid = ds_valid(:,1);

[dpsz, dpdim] = size(dp_input_train);

for i = 1: (length(dp_input_test))
    for j = 1 : dpsz
        D(i,j) = norm(dp_input_test(i,:) - dp_input_train(j,:));
    end
end

[SortedD,I] = sort(D,2);



k = [1,5,11,15,21];

for kindex = 1:5
    corr_classified_test = 0;
    for x = 1 : length(dp_input_test)
        if dp_output_test(x) * vote(x,k(kindex),I,dp_output_train) > 0
            corr_classified_test = corr_classified_test + 1;
        end
    end

    disp('when k = ');
    disp(k(kindex));
    disp('Accuracy on test set: ');
    disp(corr_classified_test/length(dp_input_test));
end

function v = vote(j,k,Index,InSet)
    p = 0;
    n = 0;
    for i = 1:k
        if InSet(Index(j,i))>0
            p = p + 1;
        else
            n = n + 1;
        end
    end
    if p / n >= 1
        v = 1;
    else
        v = -1;
    end
end

    

        
        