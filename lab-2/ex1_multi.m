%% ������������ ������ � 2: �������� ��������� ���������� ����������
%
%  ����������
%  ------------
%  ���� ��������������� ���� �� �������� �� ��������� ����� � �����������
%  �� ������������� ����� �������� ��������� � ���������� �������.
%  
%
%  ��� ������� ������ ���������� ��������� ��� � ��������� ��������:
%
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  ��� ���������� ������ ��� ������� � ��������� ������ �������� 
%  ���������� ���� ��� ��� ���������� ��������� ������������� 
%  (��� ������� �������� �������)
%

%% �������������

%% ================  1: �������� ������ ================

%% ������� ������ � �������� ����
clear all; close all; clc

fprintf('�������� ������ ...\n');

%% �������� ������
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% ����� �� ����� ����� ������ 
fprintf('������ 10 �������� �� ������ ������: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('��������� ��������������. ��� ����������� ������� enter.\n');
pause;

%% ================ 2: ���������� ��������� ================

% ��������������� ��������� � ���������� �� � �������� �������� ��������.
fprintf('���������� ��������� ...\n');

[X mu sigma] = featureNormalize(X);

% ���������� � X ���������� ������� 
X = [ones(m, 1) X];


%% ================  3: ����������� ����� � ����� ���������� alpha ================

% ====================== �������� ���� ��� ���� ======================
% ����������: ������� �������� ��� ������������ ������, ������� 
%               ���������� �������� �������� �������� �������� (alpha). 
%               
%              ���������, ��� ���������� ���� ������� - computeCostMulti � 
%               gradientDescentMulti ��������� �������� � ������ ���������� ����������. 
%               
%               ����� ��������� ����������� ����� � ������� ���������� 
%               alpha � �������� ���������. 
%
%
% ���������: � ������� ������� 'hold on' ����� ���������� ��������� ������
%            �� ����� �������.
%
%

fprintf('������ ������������ ������ ...\n');

% ������ ����� ������������ �������� alpha 
alpha = 0.01;
num_iters = 100;

% ������� ���������  Theta � �������� ����������� ����� 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% ���������� ������� ����������
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('���������� ��������');
ylabel('Cost J');

% ����� ���������� ������ ������������ ������
fprintf('Theta, ����������� ����������� ������� : \n');
fprintf(' %f \n', theta);
fprintf('\n');

%% ================  4:  ������������ ��������� �������� (����������� �����)  ========

% ������ ��������� �������� �  1650 ���������, 3 ����������

% ====================== �������� ���� ��� ���� ======================
% ���������: �������, ��� ������ ������� ������� X ������� �� ������. ����� �������,  
%            ��� ��������������� �� ����.
%
% ���������: ���������, ��� �� ��������� ������������� �������� ������

price = 0; % ��� ���� ��������


% ============================================================

fprintf(['������������� ���� �������� �  1650 ���������, 3 ���������� ' ...
         '(� �������������� ������������ ������):\n $%f\n'], price);

fprintf('��������� ��������������. ��� ����������� ������� enter.\n');
pause;

%% ================ 5: ������������� ������� ================

fprintf('������� �� ����� �������...\n');

% ====================== �������� ���� ��� ���� ======================
% ����������: ���������� � ������� normalEqn.m ��� ���������� �������  
%             ������ �������� ��������� �� ����� �������.  
%
%

%% �������� ������
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% ���������� ���������� ������� � X
X = [ones(m, 1) X];

% ���������� ���������� �� ����� �������
theta = normalEqn(X, y);

% ����� ����������� �������� �� ����� �������
fprintf('Theta ����������� �� ����� �������: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%% ================  6:  ������������ ��������� �������� (����� �������)  ========

% ������ ��������� �������� �  1650 ���������, 3 ����������

% ====================== �������� ���� ��� ����======================
price = 0; % ��� ���� ��������


% ============================================================

fprintf(['������������� ���� �������� �  1650 ���������, 3 ���������� ' ...
         '(� �������������� ����� �������):\n $%f\n'], price);

