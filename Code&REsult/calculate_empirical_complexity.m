%% ============================================
%  FAST Empirical Complexity Metrics Calculator
%  Optimized: Manual KNN (no toolbox dependency)
% ============================================
clear; clc; close all;
clear classes;  % Clear cached classes to avoid shadowing issues

datasets = {'ionospher', 'crx', 'australian', 'heart', 'wpbc', 'wdbc'};
base_path = 'C:\Users\Administrator\Desktop\EngeeneringRe03\datasets';

results = table('Size', [numel(datasets), 6], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Dataset', 'FS_Params', 'Total_FLOPs_e9', 'FS_Time_s', 'Latency_Full_ms', 'Latency_Sel_ms'});

for ds_idx = 1:numel(datasets)
    ds_name = datasets{ds_idx};
    fprintf('=== Processing Dataset: %s ===\n', ds_name);
    
    % Load Data
    try
        x = load(fullfile(base_path, [ds_name, 'Sampels.mat'])); X = x.X;
        y = load(fullfile(base_path, [ds_name, 'Label.mat']));   y = y.X;
    catch
        x = load(fullfile(base_path, [ds_name, 'Samples.mat'])); X = x.X;
        y = load(fullfile(base_path, [ds_name, 'Label.mat']));   y = y.X;
    end
    
    % Remove zero-variance features
    varX = var(X);
    zeroVarIdx = varX == 0;
    if any(zeroVarIdx)
        X(:, zeroVarIdx) = [];
    end
    
    rmed = median(X(:));
    [n, p] = size(X);
    
    % BPSO Parameters
    popSize = 80;
    maxIter = 60;
    alpha   = 0.15;
    w_start = 0.9; w_end = 0.4;
    c1 = 2.0; c2 = 2.0;
    
    pop = randi([0 1], popSize, p);
    vel = zeros(popSize, p);
    fitness = zeros(popSize,1);
    
    % Compute f2_full once
    f2_full = discriminativeScore_fast(X, y, rmed);
    
    for i = 1:popSize
        fitness(i) = objectiveFunction_fast(pop(i,:), X, y, rmed, alpha, f2_full);
    end
    pbest = pop; pbestVal = fitness;
    [gbestVal, idx] = min(fitness);
    gbest = pop(idx,:);
    
    % Main BPSO Loop with FLOPs tracking
    total_flops = 0;
    tic;
    
    hWait = waitbar(0, sprintf('Optimizing %s...', ds_name));
    
    for iter = 1:maxIter
        w_inertia = w_start - (w_start - w_end) * (iter - 1) / (maxIter - 1);
        for i = 1:popSize
            vel(i,:) = w_inertia*vel(i,:) + c1*rand(1,p).*(pbest(i,:)-pop(i,:)) + c2*rand(1,p).*(gbest-pop(i,:));
            pop(i,:) = double(1./(1+exp(-vel(i,:))) > rand(1,p));
            
            if sum(pop(i,:)) < 3
                [~, idxs] = sort(var(X,0,1), 'descend');
                pop(i, idxs(1:3)) = 1;
            end
            
            p_S = sum(pop(i,:));
            total_flops = total_flops + (1.5 * n^2 * p_S + 3.0 * n^2);
            
            fitness(i) = objectiveFunction_fast(pop(i,:), X, y, rmed, alpha, f2_full);
            if fitness(i) < pbestVal(i)
                pbest(i,:) = pop(i,:);
                pbestVal(i) = fitness(i);
            end
        end
        [val, idx] = min(fitness);
        if val < gbestVal
            gbestVal = val;
            gbest = pop(idx,:);
        end
        
        waitbar(iter/maxIter, hWait, sprintf('Iter %d/%d - %s', iter, maxIter, ds_name));
    end
    fs_time = toc;
    close(hWait);
    
    selFeatures = find(gbest == 1);
    
    % === Inference Latency (Manual KNN - No Toolbox Dependency) ===
    fprintf('Measuring inference latency (manual KNN)...\n');
    k_neighbors = 3;
    num_trials = 50;
    
    % Full features latency
    time_full = 0;
    for t = 1:num_trials
        tic;
        preds_full = manual_knn_predict(X, y, X, k_neighbors);
        time_full = time_full + toc;
    end
    latency_full_ms = (time_full / num_trials) / n * 1000;
    
    % Selected features latency
    X_sel = X(:, selFeatures);
    time_sel = 0;
    for t = 1:num_trials
        tic;
        preds_sel = manual_knn_predict(X_sel, y, X_sel, k_neighbors);
        time_sel = time_sel + toc;
    end
    latency_sel_ms = (time_sel / num_trials) / n * 1000;
    
    % Store results
    results.Dataset(ds_idx) = ds_name;
    results.FS_Params(ds_idx) = 1;
    results.Total_FLOPs_e9(ds_idx) = total_flops / 1e9;
    results.FS_Time_s(ds_idx) = fs_time;
    results.Latency_Full_ms(ds_idx) = latency_full_ms;
    results.Latency_Sel_ms(ds_idx) = latency_sel_ms;
    
    fprintf('Done. FLOPs: %.2f x 10^9, Time: %.2f s\n', total_flops/1e9, fs_time);
    fprintf('Latency Full: %.4f ms, Latency Sel: %.4f ms\n\n', latency_full_ms, latency_sel_ms);
end

disp('=== Empirical Complexity Metrics ===');
disp(results);

% Save results
fid = fopen('Complexity_Results.txt', 'w');
fprintf(fid, 'Dataset\tFS Params\tTotal FLOPs (x10^9)\tFS Time (s)\tLatency Full (ms)\tLatency Sel (ms)\n');
for i = 1:height(results)
    fprintf(fid, '%s\t%d\t%.2f\t%.2f\t%.4f\t%.4f\n', ...
        results.Dataset{i}, results.FS_Params(i), results.Total_FLOPs_e9(i), ...
        results.FS_Time_s(i), results.Latency_Full_ms(i), results.Latency_Sel_ms(i));
end
fclose(fid);
fprintf('Results saved to Complexity_Results.txt\n');

%% === FAST Helper Functions (Vectorized) ===
function f = objectiveFunction_fast(w, X, y, rmed, alpha, f2_full)
    if sum(w) < 3
        f = inf; return;
    end
    Xs = X(:, logical(w));
    f2 = discriminativeScore_fast(Xs, y, rmed);
    f_tilde = f2 / (f2_full + 1e-6);
    f = alpha*sum(w) - (1-alpha)*f_tilde;
end

function f2 = discriminativeScore_fast(X, y, rmed)
    [n,~] = size(X);
    Xc = X - rmed;
    
    % FAST: Use pdist (returns vector, not matrix)
    D_vec = pdist(Xc, 'euclidean');
    d0 = median(D_vec);
    if d0 == 0, d0 = 1e-6; end
    beta = 4 / d0;
    S_vec = 1 ./ (1 + exp(beta*(D_vec - d0)));
    
    % FAST: Vectorized index computation (no nested loops)
    k = 1:length(D_vec);
    i = ceil((2*n - 1 - sqrt((2*n-1)^2 - 8*k)) / 2);
    j = k - (i-1).*(2*n - i)/2 + i;
    
    same_class = (y(i) == y(j));
    
    if sum(~same_class) == 0 || sum(same_class) == 0
        f2 = 0;
    else
        f2 = mean(S_vec(~same_class)) - mean(S_vec(same_class));
    end
end

%% === Manual KNN Implementation (No Toolbox Required) ===
function preds = manual_knn_predict(X_train, y_train, X_test, k)
    % Compute pairwise distances between test and train
    D = pdist2(X_test, X_train);  % size: n_test x n_train
    
    % Find k nearest neighbors for each test sample
    [~, idx] = sort(D, 2, 'ascend');
    idx_k = idx(:, 1:k);
    
    % Get labels of k nearest neighbors
    labels_k = y_train(idx_k);
    
    % Majority voting for each test sample
    n_test = size(X_test, 1);
    preds = zeros(n_test, 1);
    unique_labels = unique(y_train);
    
    for i = 1:n_test
        % Count occurrences of each label
        counts = zeros(length(unique_labels), 1);
        for j = 1:length(unique_labels)
            counts(j) = sum(labels_k(i,:) == unique_labels(j));
        end
        % Assign the label with maximum count
        [~, max_idx] = max(counts);
        preds(i) = unique_labels(max_idx);
    end
end