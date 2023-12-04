close
clear 
clc

% Leitura do arquivo de entrada adaptado ao Matlab
data = readmatrix('lse_training_input.csv');

% Vetor coluna das classificações: +1 se OK e -1 se NOK
y = data(:,1);

% Matriz de parâmetros de entrada
D = data(:,[2:7]);

% Cálculo do vetor de classificação
[w] = (D'*D)^-1*D'*y;
