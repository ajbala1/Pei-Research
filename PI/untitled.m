function [outputArg1,outputArg2] = untitled(inputArg1,inputArg2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
x = [1 2; 3 4];
x(1) = [];
disp(x);
disp(length(x));
fprintf("hello world");

outputArg1 = inputArg1;
outputArg2 = inputArg2;
end