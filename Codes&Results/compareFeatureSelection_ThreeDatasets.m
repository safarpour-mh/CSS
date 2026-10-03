function compareFeatureSelection_Q1_ThreeDatasets()
%COMPAREFEATURESELECTION__THREEDATASETS
% Self-contained version: all helper functions included.
% Fixed: struct field names + function visibility.

    clc; close all; rng(1);
    
    baseDir = 'C:\Users\Administrator\Desktop\Datasets\';
    
    X1 = load(fullfile(baseDir, 'wdbcSampels.mat')).X;
    y1 = load(fullfile(baseDir, 'wdbcLabel.mat')).X;
    s1 = [2, 23, 26];
    
    X2 = load(fullfile(baseDir, 'wpbcSampels.mat')).X;
    y2 = load(fullfile(baseDir, 'wpbcLabel.mat')).X;
    s2 = [4, 10, 11, 19, 21, 27];
    
    X3 = load(fullfile(baseDir, 'heartSampels.mat')).X;
    y3 = load(fullfile(baseDir, 'heartLabel.mat')).X;
    s3 = [9, 11, 13];
    
    datasets = {
        X1, y1, s1, 'WDBC';
        X2, y2, s2, 'WPBC';
        X3, y3, s3, 'Heart'
    };

    methods_list = {'Fisher Score', 'Mutual Information', 'ReliefF', 'mRMR', 'SVM-RFE', 'L1-SVM'};
    all_results = struct();
    cleanName = @(name) regexprep(name, '[^a-zA-Z0-9_]', '_');

    for ds_idx = 1:size(datasets, 1)
        X = datasets{ds_idx, 1};
        y = datasets{ds_idx, 2};
        S = datasets{ds_idx, 3};
        ds_name = datasets{ds_idx, 4};

        fprintf('\n\n===== Processing Dataset: %s =====\n', ds_name);

        if ~iscategorical(y)
            y = categorical(y);
        end

        k = numel(S);
        cv = cvpartition(y, 'KFold', 5);
        results = [];
        selectedFeatures = struct();
        raw_accuracies = {};

        for m = 1:numel(methods_list)
            method = methods_list{m};
            clean_method = cleanName(method);
            fprintf('\n--- %s (k = %d) ---\n', method, k);
            
            switch method
                case 'Fisher Score'
                    featIdx = local_fisherScoreSelect(X, y, k);
                case 'Mutual Information'
                    featIdx = local_mutualInfoSelect(X, y, k);
                case 'ReliefF'
                    featIdx = local_relieffSelect(X, y, k);
                case 'mRMR'
                    featIdx = local_mrmrSelect(X, y, k);
                case 'SVM-RFE'
                    featIdx = local_svmRFESelect(X, y, k);
                case 'L1-SVM'
                    featIdx = local_l1SVMSelect(X, y, k);
            end
            
            selectedFeatures.(clean_method) = featIdx;
            [accMean, rawAccVec] = local_evaluateAndGetRawAccuracies(X(:, featIdx), y, cv);
            results = [results; {method, accMean}];
            raw_accuracies{end+1} = rawAccVec;
            fprintf('Mean Accuracy: %.4f\n', accMean);
        end

        clean_proposed = cleanName('Proposed');
        selectedFeatures.(clean_proposed) = S;
        fprintf('\n--- Proposed Method (k = %d) ---\n', k);
        [accProp, rawAccProp] = local_evaluateAndGetRawAccuracies(X(:, S), y, cv);
        results = [results; {'Proposed', accProp}];
        raw_accuracies{end+1} = rawAccProp;
        fprintf('Mean Accuracy: %.4f\n', accProp);

        T = cell2table(results, 'VariableNames', {'Method', 'MeanAccuracy'});
        T_sorted = sortrows(T, 'MeanAccuracy', 'descend');
        all_results.(ds_name).summary = T_sorted;
        all_results.(ds_name).raw = raw_accuracies;
        all_results.(ds_name).methods = [methods_list, {'Proposed'}];
        all_results.(ds_name).cleanMethods = [cellfun(cleanName, methods_list, 'UniformOutput', false), {clean_proposed}];

        disp(['\n=== Final Results for ' ds_name ' ===']);
        disp(T_sorted);
    end

    fprintf('\n\n===== Wilcoxon Signed-Rank Tests (Proposed > Baseline?) =====\n');
    alpha = 0.05;

    for ds_idx = 1:size(datasets, 1)
        ds_name = datasets{ds_idx, 4};
        fprintf('\n--- Dataset: %s ---\n', ds_name);
        
        raw_data = all_results.(ds_name).raw;
        clean_methods = all_results.(ds_name).cleanMethods;
        proposed_idx = find(strcmp(clean_methods, cleanName('Proposed')));
        proposed_acc = raw_data{proposed_idx};

        for m = 1:length(methods_list)
            baseline_acc = raw_data{m};
            [p, ~] = signrank(proposed_acc, baseline_acc, 'Tail', 'right');
            if p < alpha
                sig_str = ' * (significant)';
            else
                sig_str = '';
            end
            fprintf('Proposed vs %-18s: p = %.4f%s\n', methods_list{m}, p, sig_str);
        end
    end
end

% ==================== HELPER FUNCTIONS (LOCAL) ====================

function [meanAcc, rawAccuracies] = local_evaluateAndGetRawAccuracies(X, y, cv)
    classifiers = {'KNN', 'SVM', 'RF'};
    allRawAcc = [];
    for c = 1:numel(classifiers)
        modelType = classifiers{c};
        for fold = 1:cv.NumTestSets
            Xtr = X(training(cv, fold), :); ytr = y(training(cv, fold));
            Xte = X(test(cv, fold), :);   yte = y(test(cv, fold));
            switch modelType
                case 'KNN'
                    mdl = fitcknn(Xtr, ytr, 'NumNeighbors', 3);
                case 'SVM'
                    mdl = fitcsvm(Xtr, ytr, 'KernelFunction', 'rbf', 'Standardize', true);
                case 'RF'
                    mdl = TreeBagger(100, Xtr, ytr, 'Method', 'classification');
            end
            if strcmp(modelType, 'RF')
                ypred = categorical(string(predict(mdl, Xte)), categories(y));
            else
                ypred = predict(mdl, Xte);
                if ~iscategorical(ypred), ypred = categorical(ypred, categories(y)); end
            end
            foldAcc = mean(ypred == yte);
            allRawAcc(end+1) = foldAcc;
        end
    end
    meanAcc = mean(allRawAcc);
    rawAccuracies = allRawAcc(:);
end

function featIdx = local_fisherScoreSelect(X, y, k)
    classLabels = categories(y);
    nFeat = size(X,2);
    fisherScores = zeros(nFeat,1);
    for f = 1:nFeat
        featVals = X(:,f);
        overallMean = mean(featVals);
        between = 0; within = 0;
        for c = classLabels'
            idx = (y == c);
            n_c = sum(idx);
            if n_c < 2, continue; end
            classMean = mean(featVals(idx));
            classVar = var(featVals(idx), 1);
            between = between + n_c * (classMean - overallMean)^2;
            within = within + (n_c - 1) * classVar;
        end
        fisherScores(f) = (within > eps) * (between / (within + eps));
    end
    [~, idx] = sort(fisherScores, 'descend');
    featIdx = idx(1:k);
end

function featIdx = local_mutualInfoSelect(X, y, k)
    miScores = zeros(size(X,2),1);
    for f = 1:size(X,2)
        miScores(f) = local_mutualInfo_kNN(X(:,f), y, 3);
    end
    [~, idx] = sort(miScores, 'descend');
    featIdx = idx(1:k);
end

function featIdx = local_relieffSelect(X, y, k)
    y_num = double(y);
    numNeighbors = min(10, size(X,1) - 1);
    [idx, ~] = relieff(X, y_num, numNeighbors);
    featIdx = idx(1:k);
end

function featIdx = local_mrmrSelect(X, y, k)
    idx = fscmrmr(X, y);
    featIdx = idx(1:k);
end

function featIdx = local_svmRFESelect(X, y, k)
    X_std = zscore(X);
    nFeat = size(X,2);
    remaining = 1:nFeat;
    ranking = zeros(nFeat,1);
    rankIter = nFeat;
    while numel(remaining) > k
        mdl = fitcsvm(X_std(:,remaining), y, 'KernelFunction', 'linear', 'Standardize', false);
        w = mdl.Beta;
        if mdl.ClassNames(1) == mdl.Y(1)
            w = -w;
        end
        [~, sortedIdx] = sort(abs(w), 'ascend');
        toRemove = sortedIdx(1);
        ranking(remaining(toRemove)) = rankIter;
        remaining(toRemove) = [];
        rankIter = rankIter - 1;
    end
    for i = 1:numel(remaining)
        ranking(remaining(i)) = rankIter - i + 1;
    end
    [~, featIdx] = sort(ranking, 'descend');
    featIdx = featIdx(1:k);
end

function featIdx = local_l1SVMSelect(X, y, k)
    if numel(categories(y)) ~= 2
        error('L1-SVM requires binary classification.');
    end
    cats = categories(y);
    y_enc = double(y == cats(2)) * 2 - 1;
    mdl = fitclinear(X, y_enc, 'Learner', 'svm', 'Regularization', 'lasso', ...
                     'Lambda', 'auto', 'Verbose', 0);
    [~, idx] = sort(abs(mdl.Beta), 'descend');
    featIdx = idx(1:k);
end

function mi = local_mutualInfo_kNN(x, y, k_nn)
x = x(:); y = categorical(y(:)); n = length(x);
if n < k_nn + 1, k_nn = max(1, floor(n/2)); end
distX = pdist2(x, x);
[~, idxX] = sort(distX, 2);
epsX = distX(sub2ind([n,n], (1:n)', idxX(:,k_nn+1)));
classes = categories(y);
Hx_given_y = 0;
for c = classes'
    mask = (y == c); nc = sum(mask);
    if nc < 2, continue; end
    xc = x(mask);
    distXc = pdist2(xc, xc);
    if nc <= k_nn
        local_ent = log(nc);
    else
        [~, idxXc] = sort(distXc, 2);
        epsXc = distXc(sub2ind([nc,nc], (1:nc)', idxXc(:,k_nn+1)));
        nx = sum(epsXc <= (epsX(mask) + eps), 2);
        local_ent = mean(log(nx + eps));
    end
    Hx_given_y = Hx_given_y + (nc/n) * local_ent;
end
mi = log(n) - Hx_given_y;
end