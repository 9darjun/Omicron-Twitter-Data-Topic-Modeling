clc, clear, close all;
rng default
global initial_flag
%% Optimization with Genetic Algorithm
initial_flag = 0;
options = optimoptions('ga', 'PlotFcn', {@gaplotbestf, @gaplotdistance});
for i = 1:15
initial_flag = 0;
% 2d Shifted Rotated Ackley2s Function with Global Optimum on Bounds
%[gaX, gaVal, gaExitFlag, gaOutput] = ga(@(x)benchmark_func(x, 7),2,options);
% 10 d Shifted Rotated Ackley2s Function with Global Optimum on Bounds
%[gaX, gaVal, gaExitFlag, gaOutput] = ga(@(x)benchmark_func(x, 7),10,options);
% 2d Shifted Schwefel's Problem 1.2
%[gaX, gaVal, gaExitFlag, gaOutput] = ga(@(x)benchmark_func(x, 2),2,options);
% 10d Shifted Schwefel's Problem 1.2
[gaX, gaVal, gaExitFlag, gaOutput] = ga(@(x)benchmark_func(x, 2),10,options);
gaMainX(i, :) = gaX;
gaMainVal(i) = gaVal;
gaMainExitFlag(i) = gaExitFlag;
gaMainOutput(i) = gaOutput;
fname = sprintf('plot_%d.jpg', i);
saveas(gcf, fname, 'jpg')
end
gaValMax = max(gaMainVal)
gaValmin = min(gaMainVal)
gaValMean = mean(gaMainVal)
gaValStd = std(gaMainVal)