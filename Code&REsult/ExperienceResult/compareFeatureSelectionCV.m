function compareFeatureSelectionCV(X, y, S)
%COMPAREFEATURESELECTIONCV Compare classification performance with selected features
%   Stores results (table + 4 plots) in the "Results" folder.

% ---------- Input validation ----------
assert(isnumeric(X) && ismatrix(X), 'X must be a numeric matrix.');
assert(isnumeric(y) && isvector(y), 'y must be a numeric vector.');
y = y(:);
assert(all(isnumeric(S)) && all(S > 0) && all(S <= size(X,2)), ...
    'S must contain valid feature indices.');

k = 5; % 5-fold cross-validation

% ---------- Prepare output folder ----------
outDir = fullfile(pwd, 'Results');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

% ---------- Classifiers ----------
classifiers = {
    'KNN', @(Xtr,ytr) fitcknn(Xtr, ytr, 'NumNeighbors', 3, 'Standardize', true);
    'SVM', @(Xtr,ytr) fitcsvm(Xtr, ytr, 'KernelFunction', 'rbf', 'Standardize', true);
    'RF',  @(Xtr,ytr) fitcensemble(Xtr, ytr, 'Method', 'Bag', ...
                                   'Learners', templateTree('Reproducible', true), ...
                                   'NumLearningCycles', 100);
};

numClassifiers = size(classifiers, 1);
uy = unique(y);
numClasses = length(uy);

results = [];

cvp = cvpartition(y, 'KFold', k);

for clfIdx = 1:numClassifiers
    name = classifiers{clfIdx, 1};
    trainFcn = classifiers{clfIdx, 2};

    acc_o = zeros(k,1); acc_p = zeros(k,1);
    prec_o = zeros(k,1); rec_o = zeros(k,1); f1_o = zeros(k,1);
    prec_p = zeros(k,1); rec_p = zeros(k,1); f1_p = zeros(k,1);

    allFpr = linspace(0,1,100);
    allTpr_o = zeros(length(allFpr), k);
    allTpr_p = zeros(length(allFpr), k);

    for f = 1:k
        trIdx = training(cvp, f);
        teIdx = test(cvp, f);

        Xtr = X(trIdx,:); ytr = y(trIdx);
        Xte = X(teIdx,:); yte = y(teIdx);

        Xtr_p = Xtr(:, S);
        Xte_p = Xte(:, S);

        % -------- Original --------
        mdl_o = trainFcn(Xtr, ytr);
        [yp_o, sc_o] = predict(mdl_o, Xte);
        [acc_o(f), prec_o(f), rec_o(f), f1_o(f)] = metrics(yte, yp_o);

        % -------- Proposed --------
        mdl_p = trainFcn(Xtr_p, ytr);
        [yp_p, sc_p] = predict(mdl_p, Xte_p);
        [acc_p(f), prec_p(f), rec_p(f), f1_p(f)] = metrics(yte, yp_p);

        % -------- ROC Curve --------
        if numClasses == 2
            pos = uy(2);
            [fpr_o, tpr_o] = perfcurve(yte, sc_o(:,2), pos);
            [fpr_p, tpr_p] = perfcurve(yte, sc_p(:,2), pos);

            % حذف نقاط تکراری برای interp1
            [fpr_o, ia] = unique(fpr_o); tpr_o = tpr_o(ia);
            [fpr_p, ib] = unique(fpr_p); tpr_p = tpr_p(ib);

            allTpr_o(:,f) = interp1(fpr_o, tpr_o, allFpr, 'linear', 0);
            allTpr_p(:,f) = interp1(fpr_p, tpr_p, allFpr, 'linear', 0);
        end
    end

    % -------- Average Results --------
    results = [results;
        {sprintf('%s (Original)', name), mean(acc_o), mean(prec_o), mean(rec_o), mean(f1_o)};
        {sprintf('%s (Proposed)', name), mean(acc_p), mean(prec_p), mean(rec_p), mean(f1_p)}];

    % -------- Plot ROC --------
    if numClasses == 2
        figROC = figure('Name', [name ' ROC Curve'], 'Position', [100, 100, 600, 500], 'Color', 'w');
        plot(allFpr, mean(allTpr_o,2), 'b-', 'LineWidth', 1.8); hold on;
        plot(allFpr, mean(allTpr_p,2), 'r--', 'LineWidth', 1.8);
        xlabel('False Positive Rate'); ylabel('True Positive Rate');
        title([name, ' ROC Curve (Mean of ', num2str(k), '-Fold)']);
        legend('Original', 'Proposed', 'Location', 'southeast');
        grid on;
        % ذخیره نمودار ROC
        rocFile = fullfile(outDir, sprintf('%s_ROC_%s.png', name, timestamp));
        exportgraphics(figROC, rocFile, 'Resolution', 600);
        close(figROC);
    end
end

% ---------- Results Table ----------
T = cell2table(results, ...
    'VariableNames', {'Method','Accuracy','Precision','Recall','F1Score'});

disp('📊 Mean 5-Fold Classification Results:');
disp(T);

% ذخیره جدول
tableFile = fullfile(outDir, sprintf('MeanResults_%s.csv', timestamp));
writetable(T, tableFile);

% ---------- Compute Mean Accuracy for Trade-off Plot ----------
origAcc = mean([T.Accuracy(1), T.Accuracy(3), T.Accuracy(5)]);
propAcc = mean([T.Accuracy(2), T.Accuracy(4), T.Accuracy(6)]);

features = [size(X,2), numel(S)];
meanAcc = [origAcc, propAcc];

[features, idxSort] = sort(features);
meanAcc = meanAcc(idxSort);

% ---------- Plot Trade-off Chart ----------
figTrade = figure('Color', 'w', 'Position', [200, 200, 600, 400]);
plot(features, meanAcc, '-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
title('Trade-off between Number of Features and Mean Accuracy', 'FontSize', 12);
xlabel('Number of Selected Features');
ylabel('Mean Accuracy');
xticks(features);
ylim([min(meanAcc)-0.05, max(meanAcc)+0.05]);

text(features(1), meanAcc(1), sprintf('%.0f Feat (%.2f%%)', features(1), 100*meanAcc(1)), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
text(features(2), meanAcc(2), sprintf('%.0f Feat (%.2f%%)', features(2), 100*meanAcc(2)), ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');

tradeFile = fullfile(outDir, sprintf('FeatureAccuracyTradeoff_%s.png', timestamp));
exportgraphics(figTrade, tradeFile, 'Resolution', 600);
close(figTrade);

fprintf('✅ All results saved in: %s\n', outDir);

end

% ====================== زیرتوابع ======================
function [acc, prec, rec, f1] = metrics(ytrue, ypred)
    cm = confusionmat(ytrue, ypred);
    tp = diag(cm);
    fp = sum(cm,1)' - tp;
    fn = sum(cm,2) - tp;
    prec_per = tp ./ (tp + fp);
    rec_per = tp ./ (tp + fn);
    prec_per(isnan(prec_per)) = 0;
    rec_per(isnan(rec_per)) = 0;
    f1_per = 2 * (prec_per .* rec_per) ./ (prec_per + rec_per);
    f1_per(isnan(f1_per)) = 0;
    acc = sum(tp) / sum(cm(:));
    prec = mean(prec_per);
    rec = mean(rec_per);
    f1 = mean(f1_per);
end
