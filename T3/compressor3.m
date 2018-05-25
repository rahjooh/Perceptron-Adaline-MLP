function [ERROR , PSNR , OUTPUT] = compressor3(weight_1,weight_2)

file_title = [{'camera'},{'crowd'},{'house'},{'lena'},{'pepper'}];
for i = 1:5
    tsset{i} = double(imread(['TestSetHW2\TestSet\' file_title{i} '.jpg']))/255;
end

f = @(z) 1/(1+exp(-z));
PSNR = [];
for i = 1:5
    sample = tsset{i};
    Result = [];
    output = [];
    MSE_MATRIX = [];
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
            
            output(j+1,(k*64)+1:(k+1)*64) = z_2;
            ERROR = sum((z_2 - input_matrix).^2);
            Result = [Result ERROR];
        end
    end
    OUTPUT{i} = output;
    MSE_MATRIX = [MSE_MATRIX sum(Result)/numel(Result)];
    PSNR = [PSNR 10*log10(255^2/((1/256^2)*sum(sum((OUTPUT{i}-tsset{i}).^2))))];
end

ERROR = sum(MSE_MATRIX/numel(MSE_MATRIX));
PSNR = sum(PSNR)/numel(PSNR);
if  ERROR<0.05
    for i=1:numel(OUTPUT)
        OUTPUT{i} = uint8(OUTPUT{i}*255);
        imwrite(OUTPUT{i},['TEST-Compress ' file_title{i} '.jpg']);
    end
end