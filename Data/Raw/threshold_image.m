clc
clear
close all

types = {'Devanagari_', 'Western_Arabic_'};
names = {'aq', 'bc', 'cq', 'mt', 'nq', 'ra'};
threshold = 127;

for i = 1:length(types)
    for j = 1:length(names)
        fn_in = [types{i}, names{j}, '.jpg'];
        fn_out = [types{i}, names{j}, '_g.jpg'];
        im = imread(fn_in);
        d1 = im(:,:,1);
        d1 = (d1 >= threshold) * 255 + (d1 < threshold) * 0;
        im(:,:,1) = d1;
        im(:,:,2) = d1;
        im(:,:,3) = d1;
        imshow(im)
        imwrite(im, fn_out, 'jpg');
    end
end