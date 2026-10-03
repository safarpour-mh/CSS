clear; clc; close all;

%% === 1. Load Data and Preprocess ==========================
x = load('C:\Users\Administrator\Desktop\CSS_Mat-Methodology\Datasets\crxSampels.mat'); X = x.X;
y = load('C:\Users\Administrator\Desktop\CSS_Mat-Methodology\Datasets\crxLabel.mat');   y = y.X;
S=[8 9 10 11];



compareFeatureSelectionCV(X, y, S)