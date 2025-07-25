%% Cross-validated Mahalanobis Distance Analysis Script
% Author: Kelly Kadlec 
% Written: July 2021
% Last Update: July 2025
% 

%% UPDATE THIS WITH CORRECT PATH 
saveDir = 'C:\data\'; % path to data
resultsDir = 'C:\Results\'; % path to save analysis results

subject = 'n1'; % n1 or n2
win = [0.25 0.75];
itiWin = [0.25 0.75];
plotEachDay = 0; %plot individual day results
saveResults=0; %save analysis output

if strcmp(subject, 'n1')
    arrayName = {'MC', 'PPC'};
    dates = {'20200107', '20201030', '20201110', '20201113', '20201117', '20201120', '20210322'};
else
    arrayName = {'MCM', 'MCL','SPL'};
    dates = {'20230921', '20230928', '20231207', '20240116', '20240118', '20240625'};
end

results = cell(length(dates), length(arrayName));

for d = 1:length(dates)
    fprintf('Analyzing date %s (%d/%d)\n', dates{d}, d, length(dates));
    
    % Load the saved data structure
    filename = sprintf('%s_%s_analysisData.mat', subject, dates{d});
    filepath = fullfile(saveDir, filename);
   
    savedData = loadAnalysisData(filepath);
    
    if plotEachDay
    % Create figure for this date
    plt.fig('units', 'inches', 'width', 12, 'height', 9, 'font', 'Helvetica', 'fontsize', 16);
    pnl = panel();
    pnl.margin = 20;
    pnl.pack(4, 1);
    pnl.fontname = 'arial';
    pnl.fontsize = 12;
    end
    % Process each array
    for arrayIDX = 1:length(arrayName)
        
        % Build FRCell structure from saved data
        FRCell = buildFRCellFromSavedData(savedData, arrayIDX);
        
        % Get array info
        arrayData = savedData.arrays{arrayIDX};
        
        % Calculate cross-validated Mahalanobis distance
        distAll = CrossValidatedDistanceLabelNested(FRCell);
        baseLineDist = distAll{1}(:, 13);
        
        % Store results
        results{d, arrayIDX}.baseLineDist = baseLineDist;
        results{d, arrayIDX}.allDist = distAll;
        results{d, arrayIDX}.nUnits = arrayData.nUnits;
        results{d, arrayIDX}.arrayName = arrayData.arrayName;
        
        if plotEachDay
        % Plot distances
        figure;
        stem(1:13, mean(results{d, arrayIDX}.baseLineDist, 2), '.')
        set(gca, 'XTick', 1:13)
        figLabels = {savedData.metadata.conditions{:}, 'Baseline'};
        set(gca, 'XTickLabel', figLabels)
        ylabel('Distance from Baseline')
        set(gca, 'XTickLabelRotation', 45)
        title(arrayName{arrayIDX})
        end
        clear distAll baseLineDist FRCell
    end
    
    clear savedData
end

% Save results
if saveResults
    save(fullfile(resultsDir, sprintf('%s_allDays.mat', subject)), 'results');
end

%% Plot Figure 1C-G and perform signficance test

% Format data for plotting
for g = 1:13
    for j = 1:length(arrayName)
          for i = 1:length(dates)
            tmp(1, i) = nanmean(results{i, j}.baseLineDist(g));
        end
        meanResult{j}(g, 1) = mean(tmp);
        errorBar{j}(g, 1) = std(tmp) / sqrt(length(dates));
    end
end

% Plot distances averaged across days with error bar
Condition = {'Ey', 'He', 'CS', 'IS', 'CW', 'IW', 'CT', 'IT', 'CL', 'IL', 'CA', 'IA'};
colorVec = [0 1 1; 0 0.5 0.5; 0 1 0; 0 0.5 0; 1 1 0; 1 0.65 0; 1 0 0; 0.7 0 0; 1 0 1; 0.5 0 0.5; 0 0 1; 0 0 0.5];

for i = 1:length(arrayName)
    plt.fig('units', 'inches', 'width', 13, 'height', 9, 'font', 'Helvetica', 'fontsize', 32);
    pnl = panel();
    pnl.margin = 50;
    pnl.fontname = 'arial';
    pnl.fontsize = 32;
    pnl.select();
    
    for c = 1:12
        h = bar(c, meanResult{i}(c, end));
        h.FaceColor = colorVec(c, :);
        hold on
    end
    
    errorbar(1:13, meanResult{i}(:, end), errorBar{i}(:, end), 'k.');
    
    set(gca, 'XTick', 1:13)
    figLabels = {Condition{:}, 'Baseline'};
    set(gca, 'XTickLabel', figLabels)
    ylabel('Distance from Baseline')
    ylim([-0.1 35])
    
    title(sprintf('%s-%s', subject, arrayName{i}))
 
    
    hold off
    
    filename = sprintf('%s%s%s', subject, '_mahalTest-', arrayName{i});
    saveas(gcf, fullfile(resultsDir, filename), 'svg')
end

% Format data for permutaiton test
for g = 1:13
    for j = 1:length(arrayName)
        for i = 1:length(dates)
            tmp(1, i) = results{i, j}.baseLineDist(g, :);
        end
        baselineDist{j}(:, g) = tmp;
    end
end

% Permutation test for significance
for idx1 = 1:13
    for idx2 = 1:13
        for j = 1:length(arrayName)
            A = baselineDist{j}(:, idx1);
            B = baselineDist{j}(:, idx2);
            s_obs = abs(mean(A) - mean(B));
            
            % Permutation test
            s_i = [];
            Z = [A, B];
            for k = 1:10^4
                idx = randperm(12, 12);
                A_k = Z(idx(1:6));
                B_k = Z(idx(7:12));
                s_i(k) = abs(mean(A_k) - mean(B_k));
            end
            
            p = 1/k * (sum(s_i(:) > s_obs));
            sigDist{j}(idx1, idx2) = p;
        end
    end
end

% Helper function to load and verify the data structure
function savedData = loadAnalysisData(filepath)
% LOADANALYSISDATA Loads and verifies the organized data structure
%
% Input:
%   filepath - Full path to the saved .mat file
%
% Output:
%   savedData - Loaded data structure

if ~exist(filepath, 'file')
    error('File not found: %s', filepath);
end

load(filepath, 'savedData');

% Verify data structure
if ~isfield(savedData, 'analysis') || ~savedData.analysis.ready
    warning('Data structure may be incomplete or corrupted');
end

fprintf('Data loaded for subject %s, date %s\n', savedData.metadata.subject, savedData.metadata.date);
fprintf('Arrays available: ');
for i = 1:length(savedData.metadata.arrayName)
    fprintf('%s ', savedData.metadata.arrayName{i});
end
fprintf('\n');

end

function FRCell = buildFRCellFromSavedData(savedData, arrayIDX)
% BUILDFRCELL Builds FRCell structure from saved data for mahalanobis analysis
%
% Inputs:
%   savedData - Data structure created by organizeAnalysisData
%   arrayIDX - Index of the array to process (1-based)
%
% Output:
%   FRCell - Cell array organized as {condition}{position} with firing rates

arrayData = savedData.arrays{arrayIDX};
conditions = savedData.metadata.conditions;

% Initialize FRCell
FRCell = cell(1, length(conditions));

% Get all firing rates and trial info for this array
allGoFR = arrayData.firingRates.goTrials;
trialInfo = arrayData.trialInfo;

% Organize data by condition and position
for condIDX = 1:length(conditions)
    condition = conditions{condIDX};
    FRCell{condIDX} = cell(1, 5); % 5 positions
    
    for dirIDX = 1:5
        % Find trials matching this condition and position
        conditionMatch = logical([]);
        for trialIdx = 1:length(trialInfo)
            conditionMatch = [conditionMatch strcmp(trialInfo(trialIdx).Condition, condition)];
        end
        
        positionMatch = [trialInfo.PositionIDX] == dirIDX;
        trialMask = conditionMatch & positionMatch;
        
        if any(trialMask)
            % Extract firing rates for matching trials
            FRData = allGoFR(trialMask, :);
            
            % Ensure exactly 6 trials (as in original script)
            if size(FRData, 1) > 6
                % Remove excess trials (keep first 6)
                FRData = FRData(1:6, :);
            end
            
            FRCell{condIDX}{dirIDX} = FRData;
        else
            % No trials found for this condition/position combination
            FRCell{condIDX}{dirIDX} = [];
        end
    end
end

% Add baseline firing rates as the last element (as in original script)
FR_baseline = arrayData.firingRates.baseline;
FRCell{length(FRCell) + 1} = {FR_baseline};

end