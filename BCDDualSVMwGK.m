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

H2 = zeros(dpsz,dpsz);

f2 = ones(dpsz,1) * (-1);
Aeq2 = dp_output_train';
beq2 = 0;

lb2 = zeros(dpsz,1);

ite = 1;

for n = 1:9
    c = 10^(n-1);
    disp(c);
    ub2 = ones(dpsz,1) * c;
    for m = 1:5
        sigma = 10^(m-2);
        for i = 1:dpsz
            for j = 1:dpsz
                H2(i,j) = dp_output_train(j) * dp_output_train(i) * GaussianKernel(dp_input_train(j,:),dp_input_train(i,:),sigma);
            end
        end
        lamba(ite,:) = quadprog(H2,f2,[],[],Aeq2,beq2,lb2,ub2);
        sv = 331;
        for i = 1 : dpsz
            if ismembertol(lamba(ite,i),c,10^(-3))
                disp('***************************************************************************');
                sv = i;
                disp('sv found:');
                disp(sv);
                break;
            end
        end
       sum = 0;
       for i = 1 : dpsz
           sum = sum + lamba(ite,i) * dp_output_train(i) * GaussianKernel(dp_input_train(i,:),dp_input_train(sv,:),sigma);
       end
       bias2(ite) = (-1) * sum;
       
       
       corr_classified_training = 0;
       for i = 1: dpsz
            sum = 0;
            for j = 1: dpsz
                sum = sum + lamba(ite,j) * dp_output_train(j) * GaussianKernel(dp_input_train(j,:),dp_input_train(i,:),sigma);
               
            end
                
            if ( dp_output_train(i) * (sum + bias2(ite)) > 0)
               corr_classified_training = corr_classified_training + 1;        
            end
       end
	 
       corr_classified_valid = 0;
       for i = 1: length(dp_input_valid)
           sum = 0;
           for j = 1: dpsz
                sum = sum + lamba(ite,j) * dp_output_train(j) * GaussianKernel(dp_input_train(j,:),dp_input_valid(i,:),sigma);
            end
            if ( dp_output_valid(i) * (sum + bias2(ite)) > 0)
                corr_classified_valid = corr_classified_valid + 1;        
            end
        end
	
        corr_classified_test = 0;
        for i = 1: length(dp_input_test)
            sum = 0;
            for j = 1: dpsz
                sum = sum + lamba(ite,j) * dp_output_train(j) * GaussianKernel(dp_input_train(j,:),dp_input_test(i,:),sigma);
            end
            if ( dp_output_test(i) * (sum + bias2(ite)) > 0)
                corr_classified_test = corr_classified_test + 1;        
            end
        end
        
        disp('when c = ');
        disp(c);
        disp('and sigma = ');
        disp(sigma);
        disp('Accuracy on training set:');
         disp(corr_classified_training/dpsz);
        disp('Accuracy on validation set:');
        disp(corr_classified_valid/length(dp_input_valid));
        disp('Accuracy on test set:');
        disp(corr_classified_test/length(dp_input_test)); 
        ite = ite + 1;
        
    end
end

    
    




