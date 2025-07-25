%% Find shared subspaces between pairs of effectors using PLSR model
% Author: Kelly Kadlec 
% PLSM functions written by Tyson Aflalo
% Written: Jan 2022
% Last Update: July 2025
% 

dataDir='c:\data\';% path to data

subject='n1';%n1 or n2
shuffTest=2; %0 no null distribution, 1 shuffle all directions, 2 shuffle one effector only for each combination

%reps config
folds=6;
shuffReps=12; 
combReps=100;%

win=[0.25 0.75];
Condition=[{'Ey'} {'He'} {'RS'} {'LS'} {'RW'} {'LW'} {'RT'} {'LT'} {'RL'} {'LL'} {'RA'} {'LA'}];

if strcmp(subject,'n1')
    arrayName = {'MC', 'PPC'};
    load('Y:\Users\Kelly\data for Caltech data upload\n1_PLSRanalysis_data.mat')
else
    arrayName = {'MCM','MCL','SPL'};
    load('Y:\Users\Kelly\data for Caltech data upload\n2_PLSRanalysis_data.mat')
end

for arrayIDX =1:length(arrayName)
   
    FRCombo2 = saveFR{arrayIDX};
    strat_grouping = saveStratGrouping{arrayIDX};
    PositionIDX= savePositionIDX{arrayIDX};
    CondIDX=saveCondIDX{arrayIDX};
    uniqueConditions=saveUniqueConditions{arrayIDX};
    singleEffTrials=length(find(CondIDX==1));
    

    withinCorr=[];
    withinCorr_Null=[];
    %train plsm on combo and compare model performance on combo
    %effectors and non-combo effectors
    for comboIDX1=1:length(Condition)
        for comboIDX2=1:length(Condition)

            % attempting to create indicator variable matrix that
            % equates directions for effectors 1 and 5
            Within_effectors=[comboIDX1 comboIDX2];

            condtmpOrg=CondIDX==Within_effectors;
            %assuming first effector has same number of trials than
            %second effector
            numRepsEff=sum(any(condtmpOrg'));

            ef1Idx=find(condtmpOrg(:,1));
            ef2Idx=find(condtmpOrg(:,2));

            idxList=union(ef1Idx,ef2Idx);
            if(numRepsEff>singleEffTrials)
                allCombs=[];

                for i=1:combReps 
                    %numRepsEff has trials for both effectors, so we only
                    %keep half
                    cComb=[randperm(numRepsEff/2,singleEffTrials/2) randperm(numRepsEff/2,singleEffTrials/2)+singleEffTrials];
                    idCombs=idxList(cComb);
                    allCombs=vertcat(allCombs,idCombs');
                end
            else
                allCombs=[];
                %diagonal eff1=eff2
                for i=1:combReps 
                    cComb=[randperm(numRepsEff,singleEffTrials/2)];  %numRepsEff has only the trials for one effector                  
                    idCombs=[idxList(cComb);setxor(idxList(cComb),ef1Idx)]; %first half are normal trials, second half are shuffled trials
                    allCombs=vertcat(allCombs,idCombs');
                end
            end
            withinCorrComb=[];
            withinCorr_NullComb=[];
            

            for comb=1:size(allCombs,1) 

                combIdx=allCombs(comb,:);
                condTmp=condtmpOrg;
                n_ef1Idx=setdiff(ef1Idx,combIdx);
                n_ef2Idx=setdiff(ef2Idx,combIdx);
                condTmp([n_ef1Idx;n_ef2Idx],:)=0;

                condTmp=sum(condTmp,2);

                    y=zeros(size(strat_grouping,1),5);
                    for i=1:max(unique(PositionIDX))
                        y(:,i)=sum(strat_grouping==i:max(unique(PositionIDX)):length(strat_grouping),2);

                    end
                

                y(~condTmp,:)=0;
                
                %NULL distribution
                shuff_ids=find(condTmp~=0);

                if(shuffTest)
                    
                    baseIdx=(comb-1)*shuffReps;
                    halfRep=ceil(shuffReps/2);

                    if(shuffTest==2)
                        %If diagonal select half trials of effector 1 for shuffling and
                        %name them as eff 2
                        if(isempty(setxor(ef1Idx,ef2Idx)))
                            ef1Idx=combIdx(1:length(combIdx)/2);
                            ef2Idx=combIdx(length(combIdx)/2+1:end);
                        end
                    end

                    parfor nRep=1:shuffReps %PARFOR
                        y_shuff=zeros(size(y,1),size(y,2));
                        y_hat_shuff=y_shuff;
                        if(shuffTest==1)
                            y_shuff(shuff_ids,:)=y(shuff_ids(randperm(length(shuff_ids))),:);
                        elseif(shuffTest==2)

                            if(nRep>halfRep)
                                y_shuff(ef2Idx,:)=y(ef2Idx,:);
                                %shuffle first effector
                                y_shuff(ef1Idx,:)=y(ef1Idx(randperm(length(ef1Idx))),:);
                            else                                    
                                y_shuff(ef1Idx,:)=y(ef1Idx,:);
                                %shuffle second effector
                                y_shuff(ef2Idx,:)=y(ef2Idx(randperm(length(ef2Idx))),:);
                            end
                        end

                        cv_object=cvpartition(strat_grouping,'kfold',folds,'Stratify',true); %newGroups


                        y_hat=nan(size(y));

                        for fold_id=1:folds
                            train_idx=cv_object.training(fold_id);
                            test_idx=cv_object.test(fold_id);


                            goodIdx=intersect(find(train_idx), find(condTmp));

                            PLSR_tmp_shuff=FitPLSLM(FRCombo2(goodIdx,:),y_shuff(goodIdx,:),'CrossValidate',1,'Ncomp',10);
                            goodTestIdx=intersect(find(test_idx), find(condTmp));
                            X=FRCombo2(goodTestIdx,:);
                            y_hat_shuff(goodTestIdx,:)=[ones(size(X,1),1) X]*PLSR_tmp_shuff.BETA;

                        end

                        %Keep only current trials
                        goodTrials=find(condTmp);
                        y_hat=y_hat(goodTrials,:);

                        X=[];
                        PLSR_tmp=[];
                        train_idx=[];
                        test_idx=[];
                      


                        red_y_shuff=y_shuff(goodTrials,:);
                        y_hat_shuff=y_hat_shuff(goodTrials,:);
                        [tmpCorr_shuff,p]=corr(red_y_shuff(:),y_hat_shuff(:));

                        
                        withinCorr_Null(comboIDX1,comboIDX2,nRep+baseIdx)=tmpCorr_shuff;
                    end %reps
                    
                end %shuffle tests

                cv_object=cvpartition(strat_grouping,'kfold',folds,'Stratify',true); %newGroups
                y_hat=nan(size(y));
                %%%
                if(comboIDX1==comboIDX2)
                    %figure;
                    %sgtitle(['SINGLE EFFECTOR ' uniqueConditions{comboIDX1} ])
                end
                %%%%


                for fold_id=1:folds
                    train_idx=cv_object.training(fold_id);
                    test_idx=cv_object.test(fold_id);

                    %[r,c]=find(y);

                    goodIdx=intersect(find(train_idx), find(condTmp));

                    PLSR_tmp=FitPLSLM(FRCombo2(goodIdx,:),y(goodIdx,:),'CrossValidate',1);

                    goodTestIdx=intersect(find(test_idx), find(condTmp));
                    X=FRCombo2(goodTestIdx,:);
                    y_hat(goodTestIdx,:)=[ones(size(X,1),1) X]*PLSR_tmp.BETA;
                end

                %Keep only current trials
                goodTrials=find(condTmp);
                y_hat=y_hat(goodTrials,:);

                X=[];
                PLSR_tmp=[];
                train_idx=[];
                test_idx=[];
                PLSR_tmp_shuff=[];

                yred=y(goodTrials,:);
                [tmpCorr p]=(bootci(1000,@(x,y)corr(x,y),yred(:), y_hat(:)));

                withinCorrComb(comb)=nanmean(tmpCorr);

            end %combs

            withinCorr(comboIDX1,comboIDX2)=nanmean(withinCorrComb);

            
        end
       
    end

    out=withinCorr.^2;

    out_shuff=withinCorr_Null.^2;
    for comboIDX1=1:length(Condition)
        for comboIDX2=1:length(Condition)
            out_shuff_perc(comboIDX1,comboIDX2)=prctile(squeeze((out_shuff(comboIDX1,comboIDX2,:))),99);
        end
    end

    % organize effectors from head to toe
    reOrder=[out(:,1:2) out(:,10) out(:,5) out(:,12) out(:,7) out(:,11) out(:,6) out(:,9) out(:,4) out(:,8) out(:,3)];

    reOrder2=[reOrder(1:2,:); reOrder(10,:); reOrder(5 ,:); reOrder(12,:); reOrder(7 ,:); reOrder(11,:); reOrder(6 ,:); reOrder(9,:); reOrder(4 ,:); reOrder(8,:); reOrder(3 ,:)];

    results.corr{arrayIDX}=reOrder2;

    reOrderLabels={uniqueConditions{[1 2 10 5 12 7 11 6 9 4 8 3]}};

    reOrderSH=[out_shuff_perc(:,1:2) out_shuff_perc(:,10) out_shuff_perc(:,5) out_shuff_perc(:,12) out_shuff_perc(:,7) out_shuff_perc(:,11) out_shuff_perc(:,6) out_shuff_perc(:,9) out_shuff_perc(:,4) out_shuff_perc(:,8) out_shuff_perc(:,3)];

    reOrder2SH=[reOrderSH(1:2,:); reOrderSH(10,:); reOrderSH(5 ,:); reOrderSH(12,:); reOrderSH(7 ,:); reOrderSH(11,:); reOrderSH(6 ,:); reOrderSH(9,:); reOrderSH(4 ,:); reOrderSH(8,:); reOrderSH(3 ,:)];

    results.null_reps{arrayIDX}=out_shuff;
    results.null_perc{arrayIDX}=reOrder2SH;

  
    clear withinCorr withinCorr_Null withinCorrSig reOrder reOrder2
end


%% Find pairs that are signficant against null r2 and plot results
idx=0;
for c1=1:12
    for c2=1:12
        idx=idx+1;
        Combos(idx,:)=[c1,c2];
    end
end

Condition=[{'Ey'} {'He'} {'CS'} {'IS'} {'CW'} {'IW'} {'CT'} {'IT'} {'CL'} {'IL'} {'CA'} {'IA'}];

for a=1:length(arrayName)
    subtractedMatrix{a}=results.corr{a}-mean(results.null_perc{a}(:));
    sigPairs{a}=find(subtractedMatrix{a}>0);
    %which pairs share spatial coding- combo must appear on list twice to be signficant
    sigCombosAll{a}=Combos(sigPairs{a},:); 

    % Remove pairs where both effectors are the same
    validPairs = sigCombosAll{a}(sigCombosAll{a}(:,1) ~= sigCombosAll{a}(:,2), :);

    % Sort each row to treat [1 2] and [2 1] as the same pair
    sortedPairs = sort(validPairs, 2);

    % Find unique rows and their indices
    [uniquePairs, ~, idx] = unique(sortedPairs, 'rows');
    counts = accumarray(idx, 1);
    repeatedIdx = find(counts > 1);
    %Sig pairs (each sig pair will appear twice on this list)
    sigCombos{a} = uniquePairs(repeatedIdx, :);

    % Find all unique pairs from the original valid pairs (after removing [x x])
    allUniquePairs = unique(sort(validPairs, 2), 'rows');

    % Find pairs that are NOT in the repeated pairs list
    nonRepeatedPairs = setdiff(allUniquePairs, sigCombos{a}, 'rows');

    % Set matrix entries to 0 for non-repeated pairs for visualization
    for i = 1:size(nonRepeatedPairs, 1)
        row = nonRepeatedPairs(i, 1);
        col = nonRepeatedPairs(i, 2);
        subtractedMatrix{a}(row, col) = 0;
        subtractedMatrix{a}(col, row) = 0;  % Set both (i,j) and (j,i) to 0
    end

    clear counts uniquePairs repeatedIdx sortedPairs validPairs nonRepeatedPairs allUniquePairs


    %plot results
    figure;
    imagesc(tril(subtractedMatrix{a}),[0 max(subtractedMatrix{a}(:))]);colormap hot;colorbar
    set(gca,'YTick',1:12)
    set(gca,'YTickLabel',Condition,'FontSize',16)
    set(gca,'XTick',1:12)
    set(gca,'XTickLabel',Condition,'FontSize',16)
end


function [Results]=FitPLSLM(X,Y,varargin)
%% NOTE, FOR CROSAS VAL consider setting BaseLineInds so ignore prediction of baseline in performance metrics.
% Perform a glm with some extra useful stuff as optional.
% Options :
%   useGLM : whether to use poisson model GLM (otherwise a standard Linear Model)
%   CrossValidate : CrossValidated R2s
%   ShuffleTest : ShuffleTest for significance of R2
%   plotResults : whether to plot results
%   IfSig       : Does time consuming stuff only if initial pVal is signif.


[varargin,Ncomp]=ProcVarargin(varargin,'Ncomp',10);
[varargin,BaseLineInds]=ProcVarargin(varargin,'BaseLineInds',[]);

[varargin,CrossValidate]=ProcVarargin(varargin,'CrossValidate',1);
[varargin,IfSig]=ProcVarargin(varargin,'IfSig');
[varargin,NShuffles]=ProcVarargin(varargin,'NShuffles',100);
[varargin,ShuffleTest]=ProcVarargin(varargin,'ShuffleTest');
[varargin,plotResults]=ProcVarargin(varargin,'plotResults');
[varargin,GroupIDX]=ProcVarargin(varargin,'GroupIDX',[]);
[varargin,HoldOutData]=ProcVarargin(varargin,'HoldOutData',[]);
ProcVarargin(varargin);

%%
if(Ncomp>1)
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X,Y,min([size(X,2)-1,Ncomp]),'cv',7);

    [val,idx]=min(MSE(2,:));
    Results.NComp=idx;
    if Results.NComp<2
        Results.NComp=2;
    end
else
    Results.NComp=Ncomp;
end

%%
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(X,Y,Results.NComp);
yfit = [ones(size(X,1),1) X]*BETA;

Results.XS=XS;
Results.BETA=BETA;
Results.yfit=yfit;
Results.Weightings=stats.W;
Results.mu=mean(X,1);

% XS is the reduced dimensional space: To recover:
% Xshat=(X-Results.mu)*Results.Weightings;

[a,b]=corr(Y,yfit);
Results.R2=diag(a).^2;
%%

if CrossValidate
    % note that in the cross-validation procedure, I compute fit metrics
    % only on indices that are not considered "baseline"
    
 
    if ~isempty(BaseLineInds)
        if islogical(BaseLineInds); BaseLineInds=find(BaseLineInds); end
        CrossValInds=setdiff(1:size(X,1),BaseLineInds);
    else
        CrossValInds=1:size(X,1);
    end
    
    for i=CrossValInds
        FitData=X;
        TestData=FitData(i,:);
        FitData(i,:)=[];
        
        TrainTarg=Y;
        TestTarg=TrainTarg(i,:);
        TrainTarg(i,:)=[];
        
        
        trueTarg(i,:)=TestTarg;
        
        [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(FitData,TrainTarg,Results.NComp);

        if 1==0
            figure; hold on
            Yt=logical(Y);
            plot(XS(~Yt,1),XS(~Yt,2),'k.')
            
            plot(XS(Yt,1),XS(Yt,2),'r.')
        end
        %%
        testTarg(i,:)=[ones(size(TestData,1),1) TestData]*BETA;
        
        if ~isempty(HoldOutData)
            if iscell(HoldOutData)
                HoldOutPred=[ones(size(HoldOutData{1},1),1) HoldOutData{1}]*BETA;
                
                for dimIDX=1:size(trueTarg,2)
                    
                    [Results.CVR2_HO(i,dimIDX),Results.CVR2p_HO(i,dimIDX),~,Results.ErrRedCoef_HO(i,dimIDX),Results.pErrRed_HO(i,dimIDX)]=ComputeR2(HoldOutData{2}(:,dimIDX),HoldOutPred(:,dimIDX),2000);
                end
            end
        end
    end
    
    %%
    goodIDX=all(~(isnan(trueTarg) | isnan(testTarg)),2);
    if(any(goodIDX))
        [tmpCC,tmpP]=corr(trueTarg(goodIDX,:),testTarg(goodIDX,:));
        Results.CVcorr=diag(tmpCC);
        Results.CVcorrP=diag(tmpP);
    else
         Results.CVcorr=0;
         Results.CVcorrP=0;
    end
    
    for dimIDX=1:size(trueTarg,2)
        
        [Results.CVR2(dimIDX),Results.CVR2p(dimIDX),~,Results.ErrRedCoef(dimIDX),Results.pErrRed(dimIDX)]=ComputeR2(trueTarg(goodIDX,dimIDX),testTarg(goodIDX,dimIDX),2000);
    end
    
    if size(trueTarg,2)==1
        T1=logical(trueTarg);
        [Veridical,ShuffleRank,NullDist]=ShuffleCompare(testTarg(T1),testTarg(~T1),'Type','AUC');
        Results.AUCp=ShuffleRank(1);
        Results.AUC=Veridical;
        
        [Veridical,ShuffleRank,NullDist]=ShuffleCompare(testTarg(T1),testTarg(~T1),'Type','dprime');
        Results.dPp=ShuffleRank(1);
        Results.dP=Veridical;
    end
    
    Results.testTarg=testTarg;
    Results.trueTarg=trueTarg;
end

if ~isempty(GroupIDX)

    for gIDX=1:size(GroupIDX,2)
        
        idx2keep=~any(GroupIDX(:,setdiff(1:size(GroupIDX,2),gIDX)),2);
        T1=double(GroupIDX(idx2keep,gIDX));
        P1=testTarg(idx2keep);
        

        [gR2(gIDX),gR2p(gIDX)]=corr(T1,P1');
        gR2(gIDX)=sign(gR2(gIDX))* gR2(gIDX).^2;
        
        T1=logical(T1);
        [Veridical,ShuffleRank,NullDist]=ShuffleCompare(P1(T1)',P1(~T1)','Type','AUC');
        gAUCp(gIDX)=ShuffleRank(1);
        gAUC(gIDX)=Veridical;
        
        [Veridical,ShuffleRank,NullDist]=ShuffleCompare(P1(T1)',P1(~T1)','Type','dprime');
        gdPp(gIDX)=ShuffleRank(1);
        gdP(gIDX)=Veridical;
     [~,~,~,Results.(['g' num2str(gIDX) '_AUC']),AUCboot,AUCCI]=AUCTest(P1',T1,{'1' '0'});
        
    end
    
    Results.gdP=gdP;
    Results.gdPp=gdPp;
    
    Results.gAUC=gAUC;
    Results.gAUCp=gAUCp;
    Results.gR2=gR2;
    Results.gR2p=gR2p;
end
end

function [Veridical,ShuffleRank,NullDist]=ShuffleCompare(A,B,varargin)
% compare whether the mean (or median) of two distributions A and B using a shuffle test

[varargin,NShuffles]=ProcVarargin(varargin,'NShuffles',1000);
[varargin,Type]=ProcVarargin(varargin,'Type','mean');
[varargin,plotFig]=ProcVarargin(varargin,'plotFig');
ProcVarargin(varargin);
if strcmp(Type,'mean')
    Veridical=nanmean(A)-nanmean(B);
elseif strcmp(Type,'median')
    Veridical=median(A)-median(B);
elseif strcmp(Type,'AUC')
    Data=[A;B];
    Labels=[A*0;B*0+1];
    [X,Y,T,AUC,AUCboot,AUCCI]=AUCTest(Data,Labels,{'0','1'});
    Veridical=AUC;
elseif strcmp(Type,'dprime')
    Veridical=(nanmean(A)-nanmean(B))/sqrt((nanstd(A).^2+nanstd(B).^2)/2);
elseif strcmp(Type,'bhattacharyya')
    Veridical=bhattacharyya(A,B);
else
    error('Unsupported')
end
if NShuffles>0
    parfor i=1:NShuffles
        [Ashuff,~,Remaining]=Utilities.datasample([A;B],length(A),[],'Replace',false);
        Bshuff=Utilities.datasample(Remaining,[],[],'Replace',false);
        
        if strcmp(Type,'mean')
            NullDist(i)=nanmean(Ashuff)-nanmean(Bshuff);
        elseif strcmp(Type,'median')
            NullDist(i)=median(Ashuff)-median(Bshuff);
        elseif strcmp(Type,'AUC')
            Data=[Ashuff;Bshuff];
            Labels=[Ashuff*0;Bshuff*0+1];
            [X,Y,T,AUC,AUCboot,AUCCI]=AUCTest(Data,Labels,{'0','1'});
            NullDist(i)=AUC;
        elseif strcmp(Type,'dprime')
            NullDist(i)=(nanmean(Ashuff)-nanmean(Bshuff))/sqrt((nanstd(Ashuff).^2+nanstd(Bshuff).^2)/2);
        elseif strcmp(Type,'bhattacharyya')
            NullDist(i)=bhattacharyya(Ashuff,Bshuff);
        else
            error('Unsupported')
        end
    end
    
    ShuffleRank=[nnz(Veridical<NullDist)/length(NullDist),...
        nnz(Veridical==NullDist)/length(NullDist),...
        nnz(Veridical>NullDist)/length(NullDist),...
        nnz(abs(Veridical)>abs(NullDist))/length(NullDist)];
else
    NullDist=[];
    ShuffleRank=[nan nan nan nan];
end

if plotFig
%     clf
    histogram(NullDist)
    line([Veridical,Veridical], ylim, 'LineWidth',4,...
        'Color',[1 0 0])
    axis tight
    
end
end

function [X,Y,T,AUC,AUCboot,AUCCI]=AUCTest(Data,Labels,LabelNames,varargin)

% ROC analysis for binomial data - option bootstrapping of arrea under the
% curve.

[varargin,useBoot]=ProcVarargin(varargin,'useBoot');
[varargin,NSlices]=ProcVarargin(varargin,'NSlices',100);
[varargin,alpha]=ProcVarargin(varargin,'alpha',0.05);
[varargin,plotROC]=ProcVarargin(varargin,'plotROC');
ProcVarargin(varargin);
%% Process Inputs
if iscell(Data);
    NeuralDataRaw_BAK=Data; Data=[];
    Labels_BAK=Labels;Labels=[];
    
    for i=1:length(NeuralDataRaw_BAK)
        nObservations=size(NeuralDataRaw_BAK{i},1);
        Labels=[Labels;repmat(Labels_BAK(i),nObservations,1)];
        Data=[Data;NeuralDataRaw_BAK{i}];
    end
end

if length(unique(Labels))~=2; error('Only 2 Cats Supported'); end
uLabels=unique(Labels);


if length(LabelNames)==2;
    LabelNames_BAK=LabelNames; LabelNames=cell(size(Labels));
    for i=1:2
        LabelIDX=find(Labels==uLabels(i));
        LabelNames(LabelIDX)=repmat(LabelNames_BAK(i),length(LabelIDX),1);
    end
end

Labels=Labels==uLabels(1);
%% Bootstrap AUC
if useBoot
    warning off
    for i=1:NSlices
        sliceLabels=find(Labels);
        idx1 = randsample(sliceLabels,length(sliceLabels),1);
        sliceLabels=find(~Labels);
        idx2 = randsample(sliceLabels,length(sliceLabels),1);
        
        [~,~,~,AUCboot(i)]=ComputeStats2(Data([idx1;idx2]),...
            Labels([idx1;idx2]),LabelNames([idx1;idx2]));
    end
    
    AUCCI=prctile(AUCboot,[100*alpha/2 100*(1-alpha/2)]);
    warning on
else
    AUCboot=[];
    AUCCI=[];
end


[X,Y,T,AUC]=ComputeStats2(Data,Labels,LabelNames);

if plotROC
    figure; hold on
    plot(X,Y)
    xlabel('False positive rate'); ylabel('True positive rate')
    title('ROC for classification by logistic regression')
    plot([0 1],[0 1],'color',[.5 .5 .5])
end


function [X,Y,T,AUC]=ComputeStats2(Data,Labels,LabelNames)
%%
%%
% Logistic regression if Data is 
if size(Data,2)>1
    b = glmfit(Data,Labels,'Distribution','binomial','Link','logit');
    scores = mdl.Fitted.Probability;
    [X,Y,T,AUC] = perfcurve(LabelNames,scores,LabelNames{1});
else
%     if mean(Data(strcmp(LabelNames{1},LabelNames)))<mean(Data(~strcmp(LabelNames{1},LabelNames)))
%     [X,Y,T,AUC] = perfcurve(LabelNames,Data,LabelNames{end});    
%     else
    [X,Y,T,AUC] = perfcurve(LabelNames,Data,LabelNames{1});
%     end
end
end
end

function [R2,p,R2shuff,ErrRedCoef,pErrRed,MSE]=ComputeR2(X,Y,nShuffles)
[R2,SSresid]=DoIt(X,Y);

n=length(X);
MSE=sum((X-Y).^2)./n;

if nargin>2
    
    for i=1:nShuffles
        [R2shuff(i),SSresidShuff(i)]=DoIt(X,Shuffle(Y));
    end
   p=1-(nnz(R2> R2shuff)/nShuffles);
   
   ErrRedCoef=1-SSresid/mean(SSresidShuff);
   pErrRed=1-(nnz(SSresid< SSresidShuff)/nShuffles);
end


function [R2,SSresid]=DoIt(X,Y)
    SStot=sum( (X-mean(X)).^2 );
%     SSreg=sum( (Y-mean(X)).^2 );
    SSresid=sum((X-Y).^2);
    
    
    R2=1-SSresid./SStot;
end
end
   
