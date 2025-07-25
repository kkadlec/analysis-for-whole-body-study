%% Computes cross-validated mahalanobis distance 
% Author: Tyson Aflalo
function ND=CrossValidatedDistanceLabelNested(FRCell,nreps)
if nargin<2
    nreps=5000;

end
% Compute ditsance between tier 2, matching up tier 1 labels
% Note, if a tier 2 labeled set only has 1 tier 1 label, then
% assume it is baseline, and rep it to match up with all other
% tier 1 labels for the other tier 2 groups.

% geta measure of the variability of the signal. Here I am using a robust
% estimate of the std. by subtracting the mean from each group and
% estimating a pooled estimate of the std for all time points and for all
% conditions. This variability measure willbeused to scaleall distances.
%% Input
% FRCell is N Conditions x M time windows,.
% Distances are computed within and across both conditions and time.
% output is the distance

% As above, but here we are assuming that the {i,j} of FRCell
% is a cell array with each cell corresponding to a group label

%%
idx=1;
for i=1:size(FRCell,1)
    for j=1:size(FRCell,2)
        foo=cellfun(@(x)x-mean(x,1),FRCell{i,j},'UniformOutput',0);
        A{idx}=cat(1,foo{:});
        foo=cellfun(@(x)mean(x,1),FRCell{i,j},'UniformOutput',0);
        B{idx}=cat(1,foo{:});
        idx=idx+1;
    end
end


GroupWiseCov=std(cat(1,A{:}));
validIDX=GroupWiseCov~=0;
GroupWiseCov(GroupWiseCov<1)=1;
GroupWiseCov=GroupWiseCov.^2;

WeightFunction=diag(1./GroupWiseCov(validIDX));
W2=diag(WeightFunction)';
W2=repmat(W2,1,length(FRCell{1,1}));


for i=1:size(FRCell,1)
    for j=1:size(FRCell,2)
        for k=1:length(FRCell{i,j})
            FRCell{i,j}{k}=FRCell{i,j}{k}(:,validIDX);
        end
    end
end
%
Ntier1Labels=cellfun(@(x)length(x),FRCell);

if any(Ntier1Labels==1) & ~all(Ntier1Labels==1)
    baselineIDX=find(Ntier1Labels==1);

    NumGroups=max(Ntier1Labels);

    NumTrials=size( FRCell{baselineIDX}{1},1);

    SplitNumTrials=floor(NumTrials/NumGroups);

    startIDX=1;
    Data=FRCell{baselineIDX}{1};
    FRCell{baselineIDX}={};


    for i=1:NumGroups
        FRCell{baselineIDX}{1,i}=Data(startIDX:startIDX+SplitNumTrials-1,:);
        startIDX=startIDX+SplitNumTrials;
    end
end
%% Perform a cross-validated distance measure

for timeIDX1=1:size(FRCell,2)
    disp(timeIDX1)
    for timeIDX2=1:size(FRCell,2)
        for format1=1:size(FRCell,1)
            for format2=1:size(FRCell,1)
                %
                d1=FRCell{format1,timeIDX1};
                d2=FRCell{format2,timeIDX2};
                NReps=size(d1{1},1);
                %%
                for repIDX=1:nreps
                    n1=size(d1{1},1);
                    n2=size(d2{1},1);


                    foo=randperm(min([n1 n2]));

                    testIDX=foo(1:2); % two random trials
                    trainIDX=foo(3:4); % two different random trials


                    %%
                    A1=cellfun(@(x)x(trainIDX(1),:),d1,'UniformOutput',0);
                    A2=cellfun(@(x)x(trainIDX(2),:),d1,'UniformOutput',0);
                    B1=cellfun(@(x)x(testIDX(1),:),d2,'UniformOutput',0);
                    B2=cellfun(@(x)x(testIDX(2),:),d2,'UniformOutput',0);

                    A1=cat(1,A1{:})';
                    A2=cat(1,A2{:})';
                    B1=cat(1,B1{:})';
                    B2=cat(1,B2{:})';

                    D(repIDX)=(A1(:)-B1(:))'.*W2*(A2(:)-B2(:));

                end


                foo=mean(D);
                ND{format1,format2}(timeIDX1,timeIDX2)=sqrt(abs(foo)).*sign(foo);

            end
        end
    end
end
end
