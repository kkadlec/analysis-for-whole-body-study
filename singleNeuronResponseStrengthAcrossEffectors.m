%% Single Unit Analysis 
% Author: Kelly Kadlec 
% Written: July 2021
% Last Update: July 2025
% 

%% UPDATE THIS WITH CORRECT PATH 
saveDir = 'C:\Data\'; % path to data
resultsDir = 'C:\Results\'; % path to save analysis results

saveResults = 0;

subject = 'n1'; % n1 or n2
win = [0.25 0.75];

if strcmp(subject, 'n1')
    arrayName = {'MC', 'PPC'};
    dates = {'20200107', '20201030', '20201110', '20201113', '20201117', '20201120', '20210322'};
else
    arrayName = {'MCM', 'MCL', 'SPL'};
    dates = {'20230921', '20230928', '20231207', '20240116', '20240118', '20240625'};
end

Condition = {'Ey', 'He', 'RS', 'LS', 'RW', 'LW', 'RT', 'LT', 'RL', 'LL', 'RA', 'LA'};

% Initialize result storage
saveASF = cell(length(dates), length(arrayName));
saveBestEff = cell(length(dates), length(arrayName));
saveBestDir = cell(length(dates), length(arrayName));
savePanova = cell(length(dates), length(arrayName));
saveIsSig = cell(length(dates), length(arrayName));
savebestEff2 = cell(length(dates), length(arrayName));
saveIsSig2 = cell(length(dates), length(arrayName));
saveSigEff = cell(length(dates), length(arrayName));
saveFR = cell(length(dates), length(arrayName));
EffectorDiffScaled_Save = cell(length(dates), length(arrayName));
EffectorDiffZscore_Save = cell(length(dates), length(arrayName));

%% Main analysis loop
for d = 1:length(dates)
    fprintf('Processing date %s (%d/%d)\n', dates{d}, d, length(dates));
    
    % Load the saved data structure
    filename = sprintf('%s_%s_analysisData.mat', subject, dates{d});
    filepath = fullfile(saveDir, filename);
   
    savedData = loadAnalysisData(filepath);
    
    % Process each array
    for arrayIDX = 1:length(arrayName)
        fprintf('  Processing array %s\n', arrayName{arrayIDX});
        
        % Get array data
        arrayData = savedData.arrays{arrayIDX};
        ASF = arrayData.ASF_clean;
        nUnits = arrayData.nUnits;
        
        % Store ASF for this date/array
        saveASF{d, arrayIDX} = ASF;
        
        % Initialize storage for this array
        maxVal = zeros(nUnits, 1);
        bestEff = zeros(nUnits, 1);
        bestDirbestEff = zeros(nUnits, 1);
        nonParamPValue = zeros(nUnits, 1);
        isSig = false(nUnits, 1);
        effPerUnit = zeros(nUnits, 1);
        sortedFR = cell(nUnits, 1);
        maxVal2 = zeros(nUnits, 1);
        bestEff2 = zeros(nUnits, 1);
        isSig2 = false(nUnits, 1);
        sigEff = zeros(nUnits, 12);
        EffectorDiffScaled_unit = cell(nUnits, 1);
        EffectorDiffZscore_unit = cell(nUnits, 1);
        
        %% Process each unit
        for unit = 1:nUnits
            % Build firing rate matrix for this unit [condition x direction x trials]
            FR = buildFiringRateMatrix(savedData, arrayIDX, unit);
            
            if isempty(FR)
                continue;
            end
            
            % Calculate mean-subtracted values for pooled standard deviation
            valsMeanSubtracted = FR - mean(FR, 3);
            PooledSTD = std(valsMeanSubtracted(:));
            
            % Cross-validated best-worst analysis for each condition
            cvBW = zeros(size(FR, 1), 1);
            
            for format1 = 1:size(FR, 1)
                d1 = squeeze(FR(format1, :, :)); % [directions x trials]
                FRdiff = zeros(1000, 1);
                
                for repIDX = 1:1000
                    randIDs = randperm(6);
                    trainFR = d1(:, randIDs(1:5)); % Use first 5 trials for training
                    
                    % Find best and worst direction
                    [~, bestDir] = max(mean(trainFR, 2));
                    [~, worstDir] = min(mean(trainFR, 2));
                    
                    % Test on 6th trial
                    FRdiff(repIDX) = d1(bestDir, randIDs(6)) - d1(worstDir, randIDs(6));
                end
                cvBW(format1) = mean(FRdiff);
            end
            
            % Normalize by pooled standard deviation
            cvBW = cvBW ./ PooledSTD;
            
            % Store normalized values
            EffectorDiffScaled_unit{unit} = cvBW / max(cvBW); % Normalized to max
            EffectorDiffZscore_unit{unit} = cvBW; % Z-score normalized
            
            EffectorDiffScaled = cvBW;
            
            % Shuffle test for significance
            EffectorDiffScaledShuf = zeros(12, 1000);
            for i = 1:1000
                FRshuff = reshape(Shuffle(FR(:)'), size(FR));
                EffectorDiffFRshuff = max(mean(FRshuff, 3), [], 2) - min(mean(FRshuff, 3), [], 2);
                EffectorDiffScaledShuf(:, i) = EffectorDiffFRshuff ./ PooledSTD;
            end
            
            % Find best effector and test significance
            [maxVal(unit), bestEff(unit)] = max(EffectorDiffScaled);
            maxValShuf = max(EffectorDiffScaledShuf, [], 1);
            nonParamPValue(unit) = 1 - nnz(maxVal(unit) > maxValShuf) / length(maxValShuf);
            isSig(unit) = nonParamPValue(unit) < 0.05;
            
            % Count significant effectors
            countSigUnits = 1;
            tmpSigEff = zeros(1, 12);
            for c = 1:12
                nonParamPTemp = 1 - nnz(EffectorDiffScaled(c) > maxValShuf) / length(maxValShuf);
                if nonParamPTemp < 0.05
                    countSigUnits = countSigUnits + 1;
                    tmpSigEff(c) = 1;
                end
            end
            
            sigEff(unit, :) = tmpSigEff;
            [~, bestDirbestEff(unit)] = max(mean(FR(bestEff(unit), :, :),3), [], 2);
            bestDirbestEff(unit) = squeeze(bestDirbestEff(unit));
            effPerUnit(unit) = countSigUnits;
            
            % Create sorted FR for visualization
            meanFR = mean(FR, 3);
            sortedFR{unit} = circshift(meanFR, [0, (3 - bestDirbestEff(unit))]);
            
            % Find second best effector
            EffectorDiffScaled_temp = EffectorDiffScaled;
            EffectorDiffScaledShuf_temp = EffectorDiffScaledShuf;
            EffectorDiffScaled_temp(bestEff(unit)) = 0;
            EffectorDiffScaledShuf_temp(bestEff(unit), :) = 0;
            
            [maxVal2(unit), bestEff2(unit)] = max(EffectorDiffScaled_temp);
            maxValShuf2 = max(EffectorDiffScaledShuf_temp, [], 1);
            nonParamPValue2 = 1 - nnz(maxVal2(unit) > maxValShuf2) / length(maxValShuf2);
            isSig2(unit) = nonParamPValue2 < 0.05;
        end
        
        % Store results for this date/array
        saveBestEff{d, arrayIDX} = bestEff;
        saveBestDir{d, arrayIDX} = bestDirbestEff;
        savePanova{d, arrayIDX} = effPerUnit;
        saveIsSig{d, arrayIDX} = isSig;
        savebestEff2{d, arrayIDX} = bestEff2;
        saveIsSig2{d, arrayIDX} = isSig2;
        saveSigEff{d, arrayIDX} = sigEff;
        saveFR{d, arrayIDX} = sortedFR;
        EffectorDiffScaled_Save{d, arrayIDX} = EffectorDiffScaled_unit;
        EffectorDiffZscore_Save{d, arrayIDX} = EffectorDiffZscore_unit;
        
        fprintf('    Found %d/%d significantly tuned units\n', sum(isSig), nUnits);
    end
    
    clear savedData
end

%% Compile results structure
Results.diffNorm = EffectorDiffScaled_Save;
Results.diffZscore = EffectorDiffZscore_Save;
Results.pAnova = savePanova;
Results.bestEff = saveBestEff;
Results.sigEffs = saveSigEff;
Results.bestEff2 = savebestEff2;
Results.isSig2 = saveIsSig2;
Results.ASF = saveASF;
Results.isSig = saveIsSig;

% Save results
if saveResults
save(fullfile(resultsDir, sprintf('%s_unitAnalysis_results.mat', subject)), 'Results');
end
%% Plotting Figure 2 and Supplemental Figures 3-6
% Compile data across sessions for significantly tuned units
EffDiff = cell(1, length(arrayName));
EffDiff_z = cell(1, length(arrayName));
bestEffs = cell(1, length(arrayName));
effNum = cell(1, length(arrayName));
sigEff = cell(1, length(arrayName));

for a = 1:length(arrayName)
    % Initialize arrays for first concatenation
    EffDiff{a} = [];
    EffDiff_z{a} = [];
    bestEffs{a} = [];
    effNum{a} = [];
    sigEff{a} = [];
    
    for d = 1:length(dates)
        % Get significant units for this date/array
        sigUnits = Results.isSig{d, a};
        
        if any(sigUnits)
            % Extract data for significantly tuned units
            currentEffDiff = Results.diffNorm{d, a}(sigUnits);
            currentEffDiff_z = Results.diffZscore{d, a}(sigUnits);
            currentBestEffs = Results.bestEff{d, a}(sigUnits);
            currentEffNum = Results.pAnova{d, a}(sigUnits);
            currentSigEff = Results.sigEffs{d, a}(sigUnits, :);
            
            % Concatenate data
            EffDiff{a} = [EffDiff{a}, currentEffDiff'];
            EffDiff_z{a} = [EffDiff_z{a}, currentEffDiff_z'];
            bestEffs{a} = [bestEffs{a}, currentBestEffs'];
            effNum{a} = [effNum{a}, currentEffNum'];
            sigEff{a} = [sigEff{a}, currentSigEff'];
        end
    end
end


%% Figure 2A-E 
% Plot best effector distribution
colorVec = [0 1 1; 0 0.5 0.5; 0 1 0; 0 0.5 0; 1 1 0; 1 0.65 0; 1 0 0; 0.7 0 0; 1 0 1; 0.5 0 0.5; 0 0 1; 0 0 0.5];
Condition = {'Ey', 'He', 'CS', 'IS', 'CW', 'IW', 'CT', 'IT', 'CL', 'IL', 'CA', 'IA'};

for a = 1:length(arrayName)
    plt.fig('units', 'inches', 'width', 12, 'height', 9, 'font', 'Helvetica', 'fontsize', 32);
    pnl = panel();
    pnl.margin = 100;
    pnl.fontname = 'arial';
    pnl.fontsize = 32;
    
    temp = zeros(12, 1);
    effsCoded = unique(bestEffs{a});
    for i = 1:length(effsCoded)
        temp(effsCoded(i)) = sum(bestEffs{a} == effsCoded(i));
    end
    temp = 100 * (temp / length(bestEffs{a}));
    
    b = bar(temp, 'facecolor', 'flat');
    b.CData = colorVec;
    
    ylim([0 55]);
    ylabel('Percent of Tuned Units');
    set(gca, 'XTick', 1:12);
    set(gca, 'XTickLabel', Condition);
    
    if strcmp(subject, 'n2')
        title(sprintf('%s%s', 'RD-', arrayName{a}))
    else
        title(sprintf('%s%s', 'JJ-', arrayName{a}))
    end
   
end

% Signficance test for 2A-E

for j = 1:length(arrayName)
    for c = 1:12

        for i = 1:length(dates)

            countBest{j}(i,c)=length(find((Results.bestEff{i,j}==c)))';

        end
    end
end
for a = 1:length(arrayName)
    logBestEff_tmp=countBest{a}(:);

    groupID=ones(length(dates),1);
    for c = 2:12
        %         groupID=[groupID;c*ones(length(logBestEff{j}(:,1)),1)];
        groupID=[groupID;c*ones(length(dates),1)];
 
    end
    [p,anovatab,stat]=anova1(logBestEff_tmp,groupID)
    figure;[c,~,~,gnames] = multcompare(stat);

end

%% Figure 2F-J (and Supplemental Figure 6)
% Plot strength of response for each effector for each neuron
% sort by best effector and by number of effectors coded by each neuron

for a = 1:length(arrayName)
clear idx1 EffDiff_new bestEffs_red

[~,idx1]=sort(effNum{a}); 
bestEffs_red{a}=bestEffs{a}(idx1);
for i=1:length(idx1)
    EffDiff_new{i}=EffDiff{a}{idx1(i)};
end
[~,idx1]=sort(bestEffs_red{a});

plt.fig('units','inches','width',12,'height',9,'font','Helvetica','fontsize',32);    
imagesc(cat(2,EffDiff_new{idx1}),[0 1]);  
c=colorbar;
c.Ticks=[];
set(gca,'YTick',1:12)
set(gca,'YTickLabel',Condition);
xlabel('Neuron');
title(sprintf('%s-%s',subject,arrayName{a}));
end


%% Figure 2 K-O and Supplemental Figure 3
% Pie Charts for different effectors coded by single neuron 
% sorted by best effector ()
 cmap=colorVec;

for a =1:length(arrayName)
for c=1:12

try
    plt.fig('units','inches','width',12,'height',9,'font','Helvetica','fontsize',32);

    pnl = panel ();  pnl.margin=100; pnl.fontname='arial';pnl.fontsize=32;% 
    tmp=sigEff{a}(:,bestEffs{a}==c);
    tmpSum=sum(tmp,2); 
    tmpSum(c)=length(find(sum(tmp,1)==1));% only count neurons tuned to one effector for pie slice for c
    EffIDs=find(tmpSum~=0);
    for i = 1:length(EffIDs)
        
    ConditionTmp{i}=Condition{EffIDs(i)};
    end
    EffIDs=sort(EffIDs);
    tmpSum(tmpSum==0)=[];
    labels=compose('%d',tmpSum(:));

    useCMAP=cmap(EffIDs,:);
  p= pie(tmpSum,labels); colormap(useCMAP);


legend(ConditionTmp)
title(sprintf('%s-%s %s',subject,arrayName{a},Condition{c}));
hold off
clear tmp tmpSum ConditionTmp EffIDs labels useCMAP
catch
end

end
end


%% Plot histogram of effectors per unit (Supplemental Figure 4)
figure;
for a = 1:length(arrayName)
    plt.fig('units', 'inches', 'width', 13, 'height', 9, 'font', 'Helvetica', 'fontsize', 32);
    pnl = panel();
    pnl.margin = 50;
    pnl.fontname = 'arial';
    pnl.fontsize = 32;
    
    temp = 100 * histcounts(sum(sigEff{a})) / length(bestEffs{a});
    b = bar(temp, 'facecolor', 'flat');
    
    title(sprintf('%s-%s',subject,arrayName{a}));

   
    
    ylabel('Percent of units')
    xlabel('Number of effectors coded')
    ylim([0 100])
    xlim([0 13])
    set(gca, 'XTick', 1:12);
    
end

%% Helper function to shuffle data
function shuffled = Shuffle(data)
    shuffled = data(randperm(length(data)));
end

%% Helper function to build firing rate matrix from saved data
function FR = buildFiringRateMatrix(savedData, arrayIDX, unitIdx)
    % Build firing rate matrix [condition x direction x trials] for a specific unit
    
    arrayData = savedData.arrays{arrayIDX};
    conditions = savedData.metadata.conditions;
    
    % Get all firing rates and trial info for this array
    allGoFR = arrayData.firingRates.goTrials;
    trialInfo = arrayData.trialInfo;
    
    % Initialize FR matrix
    FR = zeros(length(conditions), 5, 6); % 12 conditions x 5 directions x 6 trials
    
    for condIDX = 1:length(conditions)
        condition = conditions{condIDX};
        
        for dirIDX = 1:5
            % Find trials matching this condition and position
            conditionMatch = logical([]);%
            for trialIdx = 1:length(trialInfo)
                conditionMatch = [conditionMatch strcmp(trialInfo(trialIdx).Condition, condition)];
            end
            
            positionMatch = [trialInfo.PositionIDX] == dirIDX;
            trialMask = conditionMatch & positionMatch;
            
            if any(trialMask)
                % Extract firing rates for this unit and matching trials
                unitFR = allGoFR(trialMask, unitIdx);
                
                % Ensure exactly 6 trials
                if length(unitFR) > 6
                    % Keep first 6 trials
                    unitFR = unitFR(1:6);
                end
                
                FR(condIDX, dirIDX, :) = unitFR;
            end
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