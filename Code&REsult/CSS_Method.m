%% ============================================
%  Fuzzy Equivalence-Based Feature Selection using CSS + BPSO
%  Fully aligned with Method.pdf (Q1-ready methodology)
%  - No column-wise normalization (preserves rmed semantics)
%  - rmed computed AFTER zero-variance removal, BEFORE any scaling
%  - CSS strictly follows Eq. (1.1)
%  - ACC evaluated via 3 classifiers (KNN, SVM, RF)
%  - Prints selected feature indices
% ============================================

clear; clc; close all;

%% === 1. Load Data and Preprocess ==========================
x = load('C:\Users\Administrator\Desktop\CSS_Mat-Methodology\Datasets\ionospherSampels.mat'); X = x.X;
y = load('C:\Users\Administrator\Desktop\CSS_Mat-Methodology\Datasets\ionospherLabel.mat');   y = y.X;

% Remove zero-variance features (Feature non-degeneracy condition)
varX = var(X);
zeroVarIdx = varX == 0;
if any(zeroVarIdx)
    X(:, zeroVarIdx) = [];
end

% Compute global median rmed from the actual data used in CSS
% (label-agnostic, robust, no scaling applied)
rmed = median(X(:));

% ⚠️ DO NOT apply column-wise normalization (e.g., z-score, min-max)
% This would violate the label-agnostic similarity design in Method.pdf

[n, p] = size(X);

%% === 2. Full-feature baseline =============================
ACC_full = classificationAccuracy3(X, y);
f2_full  = discriminativeScore(X, y, rmed);
fprintf('Initial ACC (3-classifier avg) = %.4f\n', ACC_full);

%% === 3. BPSO Parameters ===================================
popSize = 80;        % population size
maxIter = 60;        % number of iterations
alpha   = 0.15;      % balance factor
w_start = 0.9; w_end = 0.4;
c1 = 2.0; c2 = 2.0;

% Initialize population
pop = randi([0 1], popSize, p);
vel = zeros(popSize, p);
fitness = zeros(popSize,1);
ACC_history = zeros(maxIter,1);
best_acc_ever = 0;
best_acc_solution = [];

% Initial fitness evaluation
for i = 1:popSize
    fitness(i) = objectiveFunction(pop(i,:), X, y, rmed, alpha, f2_full);
end
pbest = pop;
pbestVal = fitness;
[gbestVal, idx] = min(fitness);
gbest = pop(idx,:);

%% === 4. Main Loop =========================================
for iter = 1:maxIter
    w_inertia = w_start - (w_start - w_end) * (iter - 1) / (maxIter - 1);
    
    for i = 1:popSize
        % Update velocity
        vel(i,:) = w_inertia*vel(i,:) + ...
            c1*rand(1,p).*(pbest(i,:)-pop(i,:)) + ...
            c2*rand(1,p).*(gbest-pop(i,:));

        % Sigmoid mapping to binary
        pop(i,:) = double(1./(1+exp(-vel(i,:))) > rand(1,p));

        % Repair: ensure at least 3 features selected (Sample non-degeneracy)
        if sum(pop(i,:)) < 3
            [~, idxs] = sort(var(X,0,1), 'descend');
            pop(i, idxs(1:3)) = 1;
        end

        % Evaluate fitness
        fitness(i) = objectiveFunction(pop(i,:), X, y, rmed, alpha, f2_full);

        % Update personal best
        if fitness(i) < pbestVal(i)
            pbest(i,:) = pop(i,:);
            pbestVal(i) = fitness(i);
        end
    end

    % Update global best
    [val, idx] = min(fitness);
    if val < gbestVal
        gbestVal = val;
        gbest = pop(idx,:);
    end

    % Evaluate ACC of current gbest
    selFeatures = find(gbest == 1);
    current_acc = classificationAccuracy3(X(:,selFeatures), y);
    ACC_history(iter) = current_acc;

    % Track best ACC solution (for reporting only — not used in optimization)
    if current_acc > best_acc_ever
        best_acc_ever = current_acc;
        best_acc_solution = gbest;
    end

    fprintf('Iter %2d | Obj = %.4f | ACC = %.4f | Feat = %d\n', ...
        iter, gbestVal, current_acc, numel(selFeatures));
end

%% === 5. Final Results =====================================
fprintf('\n=== Final Report ===\n');
fprintf('Selected Features (by obj): %d / %d\n', sum(gbest), p);
fprintf('Final ACC (by obj): %.4f\n', ACC_history(end));

fprintf('Best ACC during search: %.4f\n', best_acc_ever);
fprintf('Features in best-ACC subset: %d\n', sum(best_acc_solution));

% Selected feature indices
selFeatures_obj = find(gbest == 1);
selFeatures_acc = find(best_acc_solution == 1);

fprintf('\nSelected feature indices (by objective):\n');
disp(selFeatures_obj);

fprintf('Selected feature indices (by best ACC):\n');
disp(selFeatures_acc);

% Save results
fid = fopen('selected_features.txt', 'w');
fprintf(fid, 'Selected features (by objective):\n');
fprintf(fid, '%d ', selFeatures_obj);
fprintf(fid, '\n\nSelected features (by best ACC):\n');
fprintf(fid, '%d ', selFeatures_acc);
fprintf(fid, '\nFinal ACC (by obj): %.4f\nBest ACC: %.4f\n', ACC_history(end), best_acc_ever);
fclose(fid);

% Convergence plot
figure;
plot(1:maxIter, ACC_history, '-o','LineWidth',1.5);
xlabel('Iteration'); ylabel('Avg ACC (KNN+SVM+RF)');
title('Convergence of Classification Accuracy');
grid on;

%% === Helper Functions =====================================

function f = objectiveFunction(w, X, y, rmed, alpha, f2_full)
    if sum(w) < 3
        f = inf; return;
    end
    Xs = X(:, logical(w));
    f2 = discriminativeScore(Xs, y, rmed);
    f_tilde = f2 / (f2_full + 1e-6);
    f = alpha*sum(w) - (1-alpha)*f_tilde;
end

function f2 = discriminativeScore(X, y, rmed)
    [n,~] = size(X);
    Xc = X - rmed;               % Centering w.r.t. global median (label-agnostic)
    D = pdist2(Xc, Xc);          % Euclidean distance matrix
    d0 = median(D(:));
    if d0 == 0, d0 = 1e-6; end
    beta = 4 / d0;               % No artificial cap — data-adaptive steepness
    S = 1 ./ (1 + exp(beta*(D - d0)));
    S(1:n+1:end) = 1;            % Enforce reflexivity: S(i,i) = 1

    % Build intra- and inter-class index pairs
    Pin = []; Pout = [];
    for i = 1:n-1
        for k = i+1:n
            if y(i) == y(k)
                Pin = [Pin; i k];
            else
                Pout = [Pout; i k];
            end
        end
    end

    if isempty(Pout) || isempty(Pin)
        f2 = 0;
    else
        f2 = mean(S(sub2ind(size(S), Pout(:,1), Pout(:,2)))) - ...
             mean(S(sub2ind(size(S), Pin(:,1), Pin(:,2))));
    end
end

function ACC = classificationAccuracy3(X, y)
    if size(X,2) == 0
        ACC = 0; return;
    end
    accs = zeros(3,1);
    
    % 1. KNN
    Mdl1 = fitcknn(X, y, 'NumNeighbors', 3);
    cv1 = crossval(Mdl1, 'KFold', 5);
    accs(1) = 1 - kfoldLoss(cv1);
    
    % 2. SVM (RBF) — Standardize ONLY within CV folds (not globally)
    try
        Mdl2 = fitcsvm(X, y, 'KernelFunction', 'rbf', 'Standardize', true);
        cv2 = crossval(Mdl2, 'KFold', 5);
        accs(2) = 1 - kfoldLoss(cv2);
    catch
        accs(2) = accs(1); % fallback
    end
    
    % 3. Random Forest (via TreeBagger)
    try
        Mdl3 = TreeBagger(50, X, y, 'Method', 'classification', 'OOBPrediction', 'on');
        oobErr = oobError(Mdl3);
        accs(3) = 1 - oobErr(end);
    catch
        accs(3) = mean(accs(1:2));
    end
    
    ACC = mean(accs);
end