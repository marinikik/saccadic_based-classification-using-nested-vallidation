% Saccadic-Based Classification using Nested Validation
%
% Description:
% This script implements a pattern recognition model using a shallow
% neural network for classifying saccadic eye movement data.
%
% Validation Strategy:
% - External: 3-Fold Cross-Validation
% - Internal: Leave-One-Out Cross-Validation (LOO)
%   Used to select the optimal number of hidden layer neurons.
%
% Method:
% - Neural Network trained using MATLAB Neural Pattern Recognition Tool
%
% Requirements:
% - MATLAB Neural Network Toolbox
%
% Input Variables:
%   task2g12 - Matrix of data [id group var1 var2 ...]
%   task2g12_inputs  - Matrix of input features [var1 var2 ...]
%   task2g12_targets - Nx2 Matrix of target labels coded groups with 0/1 [1 0; 1 0; 0 1...]
%
% Author: M.-N. Koliaraki
% Date: Nov 2025



for ExtraExtRep=1:3
    clc;
    clearvars -except ExtraExtRep
    load(['task2g12.mat']);

    % create object for 3-fold stratified cross-validation partition
    y=task2g12(:,2);
    c=cvpartition(y,"KFold",3,"Stratify",true);

    % external 3-fold CV

    for k=1:3
        Q=c.TrainSize(k);

        trainIdx_ext=training(c,k);
        testIdx_ext=test(c,k);
        trainLbls=find(trainIdx_ext);
        testLbls=find(testIdx_ext);

        x = task2g12_inputs;
        t = task2g12_targets;

        Xtrain_ext=x(trainIdx_ext,:)'; %external train group inputs
        tTrain_ext=t(trainIdx_ext,:)'; %external train group targets
        Xtest_ext=x(testIdx_ext,:)'; %external test group inputs
        tTest_ext=t(testIdx_ext,:)'; %external test group targets


        % Choose a Training Function
        % For a list of all training functions type: help nntrain
        % 'trainlm' is usually fastest.
        % 'trainbr' takes longer but may be better for challenging problems.
        % 'trainscg' uses less memory. Suitable in low memory situations.

        trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

        ScoreMat = zeros(8,3); %matrix of scores for best accuracy and performance


        %Tuning hidden layer size
        i=1;
        hls=[2,3,[5:5:30]];

        for hlsInd=1:8
            X1 = zeros(Q,6);

            hiddenLayerSize=hls(hlsInd);

            % Create a Pattern Recognition Network
            net = patternnet(hiddenLayerSize, trainFcn);

            % Choose Input and Output Pre/Post-Processing Functions
            % For a list of all processing functions type: help nnprocess
            net.input.processFcns = {'removeconstantrows','mapminmax'};

            % Setup Division of Data for Training, Validation, Testing
            % For a list of all data division functions type: help nndivision
            net.divideFcn = 'dividerand';  % Divide data into training set only
            net.divideMode = 'sample';  % Divide up every sample
            net.divideParam.trainRatio = 70/100;
            net.divideParam.valRatio = 30/100;
            % net.divideParam.testRatio = 0/100;

            % Choose a Performance Function
            % For a list of all performance functions type: help nnperformance
            net.performFcn = 'crossentropy';  % Cross Entropy

            % Choose Plot Functions
            % For a list of all plot functions type: help nnplot
            net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
                'plotconfusion', 'plotroc'};



            for rep =  1:Q

                %Create training and validation sets
                if rep==1
                    trainInd = [2:Q];
                else if rep==Q
                        trainInd = [1:(Q-1)];
                else
                    trainInd = [1:rep-1,rep+1:Q];

                end
                end

                testInd = rep;

                xTrain = Xtrain_ext(:, trainInd);
                tTrain = tTrain_ext(:, trainInd);
                xTest = Xtrain_ext(:, testInd);
                tTest = tTrain_ext(:, testInd);


                % Train the Network
                [net,tr] = train(net,xTrain,tTrain);

                % Test the Network
                y = net(xTrain);
                e = gsubtract(tTrain,y);
                performance = perform(net,tTrain,y);
                tind = vec2ind(tTrain);
                yind = vec2ind(y);
                percentErrors = sum(tind ~= yind)/numel(tind);

                % Recalculate Training, Validation and Test Performance
                yTrain = net(xTrain);
                eTrain = gsubtract(tTrain,yTrain);
                performanceTrain = perform(net,tTrain,yTrain)
                tindTrain = vec2ind(tTrain);
                yindTrain = vec2ind(yTrain);
                percentErrorsTrain = sum(tindTrain ~= yindTrain)/numel(tindTrain);

                yTest = net(xTest);
                eTest = gsubtract(tTest,yTest);
                performanceTest = perform(net,tTest,yTest)
                tindTest = vec2ind(tTest);
                yindTest = vec2ind(yTest);
                percentErrorsTest = sum(tindTest ~= yindTest)/numel(tindTest);

                % Recalculate Training, Validation and Test Performance
                trainTargets = tTrain .* tr.trainMask{1};
                % valTargets = tVal.* tr.valMask{1};
                testTargets = tTest .* tr.testMask{1};
                trainPerformance = perform(net,trainTargets,yTrain)
                % valPerformance = perform(net,valTargets,y)
                testPerformance = perform(net,testTargets,yTest)

                % View the Network
                % view(net)

                % Plots
                % Uncomment these lines to enable various plots.
                %figure, plotperform(tr)
                %figure, plottrainstate(tr)
                %figure, ploterrhist(e)
                %figure, plotconfusion(t,y)
                %figure, plotroc(t,y)

                % Deployment
                % Change the (false) values to (true) to enable the following code blocks.
                % See the help for each generation function for more information.
                if (false)
                    % Generate MATLAB function for neural network for application
                    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
                    % tools, or simply to examine the calculations your trained neural
                    % network performs.
                    genFunction(net,'myNeuralNetworkFunction');
                    y = myNeuralNetworkFunction(x);
                end
                if (false)
                    % Generate a matrix-only MATLAB function for neural network code
                    % generation with MATLAB Coder tools.
                    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
                    y = myNeuralNetworkFunction(x);
                end
                if (false)
                    % Generate a Simulink diagram for simulation or deployment with.
                    % Simulink Coder tools.
                    gensim(net);
                end

               
                X1(rep,1)=hiddenLayerSize;
                X1(rep,2)=k;
                X1(rep,3)=yindTest;

                if tTest(1)==1
                    X1(rep,4)=1;
                else
                    X1(rep,4)=2;
                end
                X1(rep,5)=percentErrorsTest;
                X1(rep,6)=tr.best_perf;
            end
            
            ac = 1 - mean(X1(:,5));
            ScoreMat(i,:)=[hiddenLayerSize, ac, mean(X1(:,6))];
            i=i+1;
        end

        filename2 = sprintf('ScoreMat_k=%d-%d.mat',k, ExtraExtRep);
        save (filename2, 'ScoreMat');
        maxac=ScoreMat(1,2);
        minperf=ScoreMat(1,3);
        besthls=ScoreMat(1,1);

        for i=2:8
            if ScoreMat(i,2)>maxac
                maxac=ScoreMat(i,2);
                minperf=ScoreMat(i,3);
                besthls=ScoreMat(i,1);
            elseif ScoreMat(i,2)==maxac;
                if ScoreMat(i,3)<minperf
                    maxac=ScoreMat(i,2);
                    minperf=ScoreMat(i,3);
                    besthls=ScoreMat(i,1);
                end
            end
        end

        Final=[k,besthls,maxac,minperf];
        filename3=sprintf('Final%d-%d.mat',k, ExtraExtRep);
        save(filename3,'Final');

        hiddenLayerSize=besthls;

        % Create a Pattern Recognition Network
        net = patternnet(hiddenLayerSize, trainFcn);

        % Choose Input and Output Pre/Post-Processing Functions
        % For a list of all processing functions type: help nnprocess
        net.input.processFcns = {'removeconstantrows','mapminmax'};

        % Setup Division of Data for Training, Validation, Testing
        % For a list of all data division functions type: help nndivision
        net.divideFcn = 'dividerand';  % Divide data into training set only
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 30/100;
        net.divideParam.testRatio = 0/100;

        % Choose a Performance Function
        % For a list of all performance functions type: help nnperformance
        net.performFcn = 'crossentropy';  % Cross Entropy

        % Choose Plot Functions
        % For a list of all plot functions type: help nnplot
        net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
            'plotconfusion', 'plotroc'};

        % Train the Network
        [net,tr] = train(net,Xtrain_ext,tTrain_ext);

        % Test the Network
        y = net(Xtrain_ext);
        e = gsubtract(tTrain_ext,y);
        performance = perform(net,tTrain_ext,y)
        tind = vec2ind(tTrain_ext);
        yind = vec2ind(y);
        percentErrors = sum(tind ~= yind)/numel(tind);

        % Recalculate Training, Validation and Test Performance
        yTrain_ext = net(Xtrain_ext);
        eTrain_ext = gsubtract(tTrain_ext,yTrain_ext);
        performanceTrain_ext = perform(net,tTrain_ext,yTrain_ext)
        tindTrain = vec2ind(tTrain_ext);
        yindTrain = vec2ind(yTrain_ext);
        percentErrorsTrain = sum(tindTrain ~= yindTrain)/numel(tindTrain);

        yTest_ext = net(Xtest_ext);
        eTest_ext = gsubtract(tTest_ext,yTest_ext);
        performanceTest_ext = perform(net,tTest_ext,yTest_ext)
        tindTest_ext = vec2ind(tTest_ext);
        yindTest_ext = vec2ind(yTest_ext);
        percentErrorsTest_ext = sum(tindTest_ext ~= yindTest_ext)/numel(tindTest_ext);

        % Recalculate Training, Validation and Test Performance
        trainTargets = tTrain_ext .* tr.trainMask{1};
        % valTargets = tVal.* tr.valMask{1};
        testTargets = tTest_ext;
        trainPerformance = perform(net,trainTargets,yTrain_ext);
        % valPerformance = perform(net,valTargets,y)
        testPerformance = perform(net,testTargets,yTest_ext);


        % write- results
        X1=[testLbls,  tTest_ext', yindTest_ext'];
        [MX1,NX1]=size(X1);
        X1(MX1,(NX1+1))=testPerformance;
       


        filename4=sprintf('results-task2g12-k=%d fold-%d.xlsx',k, ExtraExtRep);
        xlswrite(filename4,X1);

        close all
    end

end