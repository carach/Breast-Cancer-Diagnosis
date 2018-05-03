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

H1 = [eye(dpdim),zeros(dpdim, dpsz+1)];
H1 = [H1 ; zeros( dpsz+1,dpdim + dpsz +1)];

f1 = [ zeros(dpdim+1,1); ones(dpsz,1)];
b = zeros(dpsz,1) * (-1);

A1 = [dp_input_train,ones(dpsz,1)]; % construct the first part of A
[A1rows,A1columns] = size(A1);
for m = 1:A1rows
    for n = 1:A1columns
        A1(m,n) = A1(m,n) * dp_output_train(m) * (-1);
    end
end
A2 = eye(dpsz) * (-1);
A = [A1,A2];

Aeq = [];
beq = [];

lb = [ones(dpdim + 1, 1) * (-inf);zeros( dpsz,1)];
ub = ones(dpdim + 1 + dpsz,1) * (inf);



for i = 1:9
    c = 10^(i-1);
    disp(c);
    f1 = f1 * c;
    classifier = quadprog(H1,f1,A,b,Aeq,beq,lb,ub);
    weight = classifier(1:dpdim,:);
    bias = classifier(dpdim + 1,:);
    
    corr_classified_training = 0;
    for n = 1: dpsz
        if ( dp_output_train(n) * (dp_input_train(n,:) * weight + bias) > 0)
        corr_classified_training = corr_classified_training + 1;        
        end
    end
    
    corr_classified_valid = 0;
    for n = 1: length(dp_input_valid)
        if ( dp_output_valid(n) * (dp_input_valid(n,:) * weight + bias) > 0)
        corr_classified_valid = corr_classified_valid + 1;        
        end
    end
    
    corr_classified_test = 0;
    for n = 1: length(dp_input_test)
         if ( dp_output_test(n) * (dp_input_test(n,:) * weight + bias) > 0)
         corr_classified_test = corr_classified_test + 1;        
         end
    end
   
    disp('weight:');
    disp(weight);
    disp('bias:');
    disp(bias);
    disp('Accuracy on training set:');
    disp(corr_classified_training/dpsz);
    disp('Accuracy on validation set:');
    disp(corr_classified_valid/length(dp_input_valid));
    disp('Accuracy on test set:');
    disp(corr_classified_test/length(dp_input_test)); 
       
end