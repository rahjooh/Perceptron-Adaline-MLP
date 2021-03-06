function compressionPhoto
clc
clear
alpha = 0.5;
Epp = 0;
Counter1 = 0;
MSE = Inf;
s = 0;
weight_1 = -rand(52,65)+0.5;
weight_2 = -rand(64,53)+0.5;

for i = 0:90
    TrainSet{i+1} = double(imread(['TrainSet\train (' num2str(i) ').jpg']))/255;
end



f = @(z) 1/(1+exp(-z));
f_prime = @(z) f(z)*(1-f(z));

while Epp<=30&&MSE>= 0.03    
    for i = 0:90        
        sample = TrainSet{i+1};
        for j =0:255
            for k =0:3
                input_matrix = sample(j+1,(k*64)+1:(k+1)*64);
                
                for m = 1:52
                    y_in_1(m) = sum(weight_1(m,:).*[1 input_matrix]);
                    z_1(m) = f(y_in_1(m));
                end
                
                for m = 1:64
                    y_in_2(m) = sum(weight_2(m,:).*[1 reshape(z_1,1,52)]);
                    z_2(m) = f(y_in_2(m));
                end
                
                for m = 1:64
                    delta_2(m) = (input_matrix(m) - z_2(m))*(f_prime(y_in_2(m)));
                end
                
                weight_2(:,1) = weight_2(:,1) + alpha*(delta_2)';
                for d = 2:52
                    weight_2(:,d) = weight_2(:,d) + alpha*(delta_2)'*z_1(d-1);
                end
                
                for d = 1:52
                    delta_in(d) = sum(delta_2' .* weight_2(:,d+1));
                end
                
                for d= 1:52
                    delta_1(d) = delta_in(d)*f_prime(y_in_1(d));
                end
                
                weight_1(:,1) = weight_1(:,1) + alpha*(delta_1)';
                for d = 2:64
                    weight_1(:,d) = weight_1(:,d) + (alpha*(delta_1).*input_matrix(d-1))';
                end
                
            end
        end
        
    end
    last_MSE = MSE;    
    [MSE , PSNR ,OUTPUT] = compressor_Test(weight_1,weight_2);
        
    if last_MSE == MSE||last_MSE<MSE
        Counter1 = Counter1 +1;
    else
        Counter1 = 0;
    end
    
    if Counter1 >=3
        s = 1;
        file_title = [{'camera'},{'crowd'},{'house'},{'lena'},{'pepper'}];
        for L=1:numel(OUTPUT)
            OUTPUT{L} = uint8(OUTPUT{L}*255);
            imwrite(OUTPUT{L},['TEST-Compress ' file_title{L} '.jpg']);
        end        
    end 
    
    if s ==1
        break
    end
    
    Epp = Epp+1;
    p_Result(Epp,:) = [Epp MSE];
    disp(MSE);
end
if Epp>=30
    file_title = [{'camera'},{'crowd'},{'house'},{'lena'},{'pepper'}];
    for L=1:numel(OUTPUT)
        OUTPUT{L} = uint8(OUTPUT{L}*255);
        imwrite(OUTPUT{L},['TEST-Compress ' file_title{L} '.jpg']);
    end
end
hold on
grid minor
xlabel ('Epp');
ylabel ('MSE');
title ({['MSE On Test VS Epp']; ...
    ['PSNR : ' num2str(PSNR)];...
    ['Number Of Hiden Layer Neurons : ' num2str(numel(weight_1(:,1)))]});
plot(p_Result(:,1),p_Result(:,2));