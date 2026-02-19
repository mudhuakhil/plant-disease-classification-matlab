%% Project Title    : Plant Disease Classification
%  Diseases Analyzed: Anthracnose & Blackspot
%  Author           : Manu B.N (Updated Version)

close all;
clear;
clc;

%% ------------------------------------------------------------
%  IMAGE INPUT
% -------------------------------------------------------------
[filename, pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'}, ...
                                 'Pick a Disease Affected Leaf');
I = imread(fullfile(pathname, filename));
figure; imshow(I); title('Disease Affected Leaf');

%% ------------------------------------------------------------
%  COLOR SEGMENTATION USING K-MEANS CLUSTERING
% -------------------------------------------------------------
cform = makecform('srgb2lab');
lab_he = applycform(I, cform);

ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);

ab = reshape(ab, nrows*ncols, 2);
nColors = 3;

[cluster_idx, cluster_center] = kmeans(ab, nColors, ...
                                       'distance', 'sqEuclidean', ...
                                       'Replicates', 3);

pixel_labels = reshape(cluster_idx, nrows, ncols);

segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels, [1 1 3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end

figure;
subplot(3,1,1); imshow(segmented_images{1}); title('Cluster 1');
subplot(3,1,2); imshow(segmented_images{2}); title('Cluster 2');
subplot(3,1,3); imshow(segmented_images{3}); title('Cluster 3');

%% ------------------------------------------------------------
%  FEATURE EXTRACTION
% -------------------------------------------------------------
x = inputdlg('Enter the cluster number containing the diseased region:');
i = str2double(x);

seg_img = segmented_images{i};

% Convert to grayscale if needed
if ndims(seg_img) == 3
    img = rgb2gray(seg_img);
else
    img = seg_img;
end

% GLCM Matrix
glcms = graycomatrix(img);

% GLCM Features
stats = graycoprops(glcms, 'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;

% Statistical Features
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1 - (1/(1 + a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));

% Inverse Difference Moment (IDM)
[m, n] = size(img);
in_diff = 0;
for p = 1:m
    for q = 1:n
        in_diff = in_diff + (double(img(p,q)) / (1 + (p - q)^2));
    end
end
IDM = double(in_diff);

% Feature Vector
feat_disease = [Contrast, Correlation, Energy, Homogeneity, ...
                Mean, Standard_Deviation, Entropy, RMS, Variance, ...
                Smoothness, Kurtosis, Skewness, IDM];

%% ------------------------------------------------------------
%  CLASSIFICATION USING NEW MATLAB SVM (fitcsvm)
% -------------------------------------------------------------
load Diseaseset.mat;
% Variables:
%   diseasefeat → Training features
%   diseasetype → Labels

svmModel = fitcsvm(diseasefeat, diseasetype, ...
                   'KernelFunction', 'linear', ...
                   'Standardize', true);

% Predict class
species_disease = predict(svmModel, feat_disease);

%% ------------------------------------------------------------
%  DISPLAY RESULT
% -------------------------------------------------------------
fprintf('\n=========================================\n');
fprintf('     PREDICTED DISEASE TYPE: %s\n', string(species_disease));
fprintf('=========================================\n');