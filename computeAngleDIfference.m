%% For each pair of effectors, compute a preferred direction for each and then compute the angle between them.
% Author: Kelly Kadlec 
% Written: Dec 2021
% Last Update: July 2025
% 

subject = 'n1'; % n2
win = [0.25 0.75];

%% UPDATE THIS WITH CORRECT PATH 
saveDir = 'C:\Data\';% path to data
resultsDir = 'C:\Results\';% path to save analysis results

if strcmp(subject, 'n1')
    arrayName = {'MC', 'PPC'};
    dates = {'20200107', '20201030', '20201110', '20201113', '20201117', '20201120', '20210322'};
    load(fullfile(saveDir,'n1_1hz_forAnalysis.mat'));
else
    arrayName = {'MCM', 'MCL', 'SPL'};
    dates = {'20230921', '20230928', '20231207', '20240116', '20240118', '20240625'};
    load(fullfile(saveDir,'n2_1hz_forAnalysis.mat'));
end

Condition = {'Ey', 'He', 'RS', 'LS', 'RW', 'LW', 'RT', 'LT', 'RL', 'LL', 'RA', 'LA'};


% Initialize results storage
resultsAngComp = cell(length(dates), length(arrayName));

for d = 1:length(dates)

    fprintf('Processing date %s (%d/%d)\n', dates{d}, d, length(dates));

    % Load the saved data structure
    filename = sprintf('%s_%s_analysisData.mat', subject, dates{d});
    filepath = fullfile(saveDir, filename);

    if ~exist(filepath, 'file')
        error('Saved data file not found: %s. Run data organization script first.', filepath);
    end

    savedData = loadAnalysisData(filepath);

    for arrayIDX = 1:length(arrayName)

        % Get the array data from saved structure
        arrayData = savedData.arrays{arrayIDX};
        nUnits = arrayData.nUnits;

        % Get firing rate data and trial info directly from saved structure
        allGoFR = arrayData.firingRates.goTrials; % [trials x units]
        trialInfo = arrayData.trialInfo; % Contains Condition, PositionIDX, Position for each trial

        % Apply position switching for left side body movements to account for intrinsic reference frame (n1 only)--
        % Switches for MC and then switches back to original for PPC which
        % has mixed intrinsic and extrinsic reference frames
        modifiedTrialInfo = trialInfo;
        if strcmp(subject, 'n1')
            for i = 1:length(modifiedTrialInfo)
                for j = 1:length(modifiedTrialInfo(i).Condition)
                    if any(strcmp(modifiedTrialInfo(i).Condition{j}, {'LS', 'LW', 'LT', 'LL', 'LA'}))
                        tmp = modifiedTrialInfo(i).PositionIDX(j);

                        if tmp == 1
                            modifiedTrialInfo(i).PositionIDX(j) = 3;
                            modifiedTrialInfo(i).Position{1,j} = [-0.3782 0.1 0.1302];
                        elseif tmp == 3
                            modifiedTrialInfo(i).PositionIDX(j) = 1;
                            modifiedTrialInfo(i).Position{1,j} = [0.3825 0.1 0.1169];
                        elseif tmp == 4
                            modifiedTrialInfo(i).PositionIDX(j) = 5;
                            modifiedTrialInfo(i).Position{1,j} = [0.2294 0.1 -0.3277];
                        elseif tmp == 5
                            modifiedTrialInfo(i).PositionIDX(j) = 4;
                            modifiedTrialInfo(i).Position{1,j} = [-0.2407 0.1 -0.3195];
                        elseif tmp == 2
                            modifiedTrialInfo(i).PositionIDX(j) = 2;
                        end
                    end
                end
            end
        end

        coef = fitPositionRegressionFromSavedData(savedData, arrayIDX, win);

        for unit = 1:nUnits
            for condIDX1=1:12
                for condIDX2=1:12
                    AngCmp(condIDX1,condIDX2,unit)=180/pi*acos(dot(coef(2:3,condIDX1,unit),coef(2:3,condIDX2,unit))./(norm(coef(2:3,condIDX1,unit))*norm(coef(2:3,condIDX2,unit))) );

                    tmp=cross([coef(2:3,condIDX1,unit); 0]./norm(coef(2:3,condIDX1,unit)),[coef(2:3,condIDX2,unit); 0]./norm(coef(2:3,condIDX2,unit)));
                    AngCmp(condIDX1,condIDX2,unit)=AngCmp(condIDX1,condIDX2,unit)*sign(tmp(3));
                end
            end
        end

        %Specifically look at the difference between the best and second
        %best effector coded, only looking at neurons with signficant
        %responses as determined by analysis in
        %singleNueronStrengthAcrossEffectors.m
        saveIDX=1;
        saveDiff=[];
        for unit=1:nUnits
            unitIDX=1;
            angleDiff=[];
            if Results.sigEffs{d,arrayIDX}(unit,Results.bestEff{d,arrayIDX}(unit)) && Results.sigEffs{d,arrayIDX}(unit,Results.bestEff2{d,arrayIDX}(unit))
                angleDiff=squeeze(AngCmp(Results.bestEff{d,arrayIDX}(unit),Results.bestEff2{d,arrayIDX}(unit),unit));
            end
            if ~isempty(angleDiff)
                saveDiff(saveIDX,1)=real(angleDiff);
                saveIDX=saveIDX+1;
            end
        end


        if ~isempty(angleDiff) && ~isnan(angleDiff)
            saveDiff(saveIDX, 1) = real(angleDiff);
            saveIDX = saveIDX + 1;
        end


        % Store results
        if ~isempty(saveDiff)
            resultsAngComp{d, arrayIDX} = saveDiff;
        else
            resultsAngComp{d, arrayIDX} = {};
        end

        % Clear variables for next iteration
        clear coef mdl modifiedTrialInfo allCoef
    end

    % Clear variables for next date
    clear savedData arrayData allGoFR trialInfo
end

%% Plot Figure 3 and compute statistics
for a = 1:length(arrayName)
    plt.fig('units', 'inches', 'width', 12, 'height', 9, 'font', 'Helvetica', 'fontsize', 32);
    tmpAng = [];

    % Collect all angle differences across dates
    for d = 1:length(dates)
        if ~isempty(resultsAngComp{d, a})
            if isempty(tmpAng)
                tmpAng = resultsAngComp{d, a};
            else
                tmpAng = [tmpAng; resultsAngComp{d, a}];
            end
        end
    end

    % Compute circular statistics
    if ~isempty(tmpAng)
        meanDifference{a} = circ_mean(tmpAng * pi/180) * 180/pi; % Convert to radians for calculation, back to degrees
        ciDifference{a} = circ_confmean(tmpAng * pi/180, 0.05) * 180/pi; % Same conversion

        % Create histogram
        figure;
        histogram(tmpAng(:), [-180:20:180], 'FaceColor', 'b', 'Normalization', 'probability');

        ylim([0 0.5])
        ylabel('Portion of Neurons')
        xlabel('Angle of difference between directions')
        title(sprintf('%s-%s-n=%i', subject, arrayName{a}, length(tmpAng)))
    else
        fprintf('No data available for %s array %s\n', subject, arrayName{a});
    end
end


% Helper function to load analysis data (if not already defined)
function savedData = loadAnalysisData(filepath)
% LOADANALYSISDATA Loads the organized data structure
%
% Input:
%   filepath - Full path to the saved data file
%
% Output:
%   savedData - The loaded data structure

if ~exist(filepath, 'file')
    error('Data file not found: %s', filepath);
end

loaded = load(filepath);
savedData = loaded.savedData;

fprintf('Loaded data for subject %s, date %s\n', ...
    savedData.metadata.subject, savedData.metadata.date);
end