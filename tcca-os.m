m=load('/Users/jianqinsun/Downloads/result_data/Normalized_WT_result.csv'); 
N=load('/Users/jianqinsun/Downloads/result_data/Normalized_CORR_result.csv');
L=load('/Users/jianqinsun/Downloads/result_data/BoW_int_result.csv');
label=load("/Users/jianqinsun/Downloads/result_data/label.txt");
% m = m+normrnd(0,0.3,11647,128);  % 降维后的数据
% N= N+normrnd(0,0.3,11647,144);
% L = L+normrnd(0,0.3,11647,500);

tic;
[coeff, score]= pca(m);  % coeff是主成分分量，即样本协方差矩阵的特征向量；score是主成分，即data在低维空间的投影，也就是降维后的数据，维度和data相同，若想降维到k维，只需要取前k列即可。
M1 = score(:, 1:2);  % 降维后的数据
[coeff, score]= pca(N);
N1 = score(:, 1:2); 
[coeff, score]= pca(L);
L1 = score(:, 1:2); 
% M1 = M1+normrnd(0,0.3,11647,20);  % 降维后的数据
% N1 = N1+normrnd(0,0.3,11647,20);
% L1 = L1+normrnd(0,0.3,11647,20);


% ----------------------------------------
% testRatio = 0.3;
% tic;
% CCA(m,N,100,100);
% tt1=toc;
% 
% 
% tic;
% cx=0.038;cy=0.008;
% X=M1;Y=L1;
% [A,B,r,U,V] = scanoncorr(X,Y,cx,cy);
% 
% tic;
% mdr_tcca({M1,N1,L1});
% tt5=toc;
tic;
mvdr_ttcca({M1,N1,L1});
tt6=toc;


%璁剧疆��濮������?

load('M.mat');
load('U1.mat');
load('U2.mat');
load('U3.mat');
opt.d1=2;
opt.d2=2;
opt.d3=2;
opt.d=2;
opt.kmax=2;
opt.rho=5;
opt.tau=5;
opt.lambda1=0.4;
opt.lambda2=0.3;
opt.lambda3=0.35;

%���伴����grid
  

d1=opt.d1;
d2=opt.d2;
d3=opt.d3;
d=opt.d;
kmax=opt.kmax;
rho=opt.rho;
tau=opt.tau;
lambda1=opt.lambda1;
lambda2=opt.lambda2;
lambda3=opt.lambda3;
A_0=tensor(ones(d1,d2,d3));
B1_0 =zeros(d1,d);  
B2_0 =zeros(d2,d);
B3_0 =zeros(d3,d);
P_0=tensor(zeros(d,d,d));
hatM_0=tensor(zeros(d1,d2,d3));
U1_0 =U1;
U2_0 =U2;
U3_0 =U3;
V1_0 =U1;
V2_0 =U2;
V3_0 =U3;
tic;
%%    寰���缁��� 
 for k =1:kmax
%%    P_k  ��杩�浠ｆ��
tV=kron(U3_0.',U2_0.');
tlideV=kron(tV,U1_0.');
tU=kron(V3_0.',V2_0.');
tlideU=kron(tU,V1_0.');
vP_k=(tlideV*tlideV.'+rho*ones(d^3,d^3))\(tlideV*tens2vec(double(M),1)+tlideU*((tens2vec(double(A_0),1)+rho*tens2vec(double(hatM_0),1))));
P_k=vec2tens(vP_k,[d,d,d]);
tP_k=tensor(P_k);

%%     hatM_k ��杩�浠ｆ��
vhatM_k = tlideU.'*vP_k-1/rho*tens2vec(double(A_0),1);
hatM_k  = vec2tens(vhatM_k,[d1,d2,d3]);


%%    Vp_k ��杩�浠ｆ��
V_11=kron(V3_0,V2_0);
V_22=kron(V3_0,V1_0);
V_33=kron(V2_0,V1_0);

tg1=-B1_0+rho*(V1_0-U1_0);
t1=tenmat(P_k,1);
t2=tenmat(P_k,2);
t3=tenmat(P_k,3);
m1=tenmat(M,1);
m2=tenmat(M,2);
m3=tenmat(M,3);
tff1=-2*tenmat(M,1)*V_11*t1'+V1_0*t1*V_11.'*V_11*t1';
tf1=double(tff1);
F1=tg1+tf1;

tg2=-B2_0+rho*(V2_0-U2_0);
tff2=-2*tenmat(M,2)*V_22*t2'+V2_0*t2*V_22.'*V_22*t2';
tf2=double(tff2);
F2=tg2+tf2;

tg3=-B3_0+rho*(V3_0-U3_0);
tff3=-2*tenmat(M,3)*V_33*t3'+V3_0*t3*V_33.'*V_33*t3';
tf3=double(tff3);
F3=tg3+tf3;

%prox
%�����芥�板舰寮�
 %lambda1=0.5;
     V1_k = zeros (d1,d);
    for m = 1:2
        if norm(F1(:,m)) >= 1/lambda1
           V1_k(:,m) = V1_0(:,m) - lambda1*(V1_0(:,m)/norm(V1_0(:,m)));
        else
           V1_k(:,m) = 0;
        end
    end
    
%lambda2=0.5;
     V2_k = zeros (d2,d);
    for m = 1:2
        if norm(F2(:,m)) >= 1/lambda1
           V2_k(:,m) = V2_0(:,m) - lambda1*(V2_0(:,m)/norm(V2_0(:,m)));
        else
           V2_k(:,m) = 0;
        end
    end
    
%lambda3=0.5;
     V3_k = zeros (d3,d);
    for m = 1:2
        if norm(F3(:,m)) >= 1/lambda1
           V3_k(:,m) = V3_0(:,m) - lambda1*(V3_0(:,m)/norm(V3_0(:,m)));
        else
           V3_k(:,m) = 0;
        end
    end
    
    
    
%%    Up_k ��杩�浠ｆ��
U_11=kron(U3_0,U2_0);
U_22=kron(U3_0,U1_0);
U_33=kron(U2_0,U1_0);
Q_1=double((tenmat(A_0,1)+rho*tenmat(hatM_0,1))*U_11*t1')+rho*(V1_0-B1_0);
[U,S,V]=svd(Q_1);
U=U(:,1:d);
U1_k=U*V.';
Q_2=double((tenmat(A_0,2)+rho*tenmat(hatM_0,2))*U_22*t2')+rho*(V2_0-B2_0);
[U,S,V]=svd(Q_2);
U=U(:,1:d);
U2_k=U*V.';
Q_3=double((tenmat(A_0,3)+rho*tenmat(hatM_0,3))*U_33*t3')+rho*(V3_0-B3_0);
[U,S,V]=svd(Q_3);
U=U(:,1:d);
U3_k=U*V.';

%%     u   w  s��杩�浠ｆ��

modeP= ttm(tP_k,{U1_k,U2_k,U3_k},[1,2,3]);
A_k     = double((A_0))+rho*(hatM_k-double(modeP));

B1_k    = B1_0+rho*(U1_k-V1_k);
B2_k    = B2_0+rho*(U2_k-V2_k);
B3_k    = B3_0+rho*(U3_k-V3_k);

%% termination check
err1=0.1;
err2=0.5;
pmu=frob(double(modeP)-hatM_k);
pu=frob(double(modeP)-hatM_k)+norm(U1_k,'fro')+norm(U2_k,'fro')+norm(U3_k,'fro');
prim=pmu+pu;
dmv=frob(hatM_k-double(hatM_0));
dv=frob(hatM_k)+norm(V1_k,'fro')+norm(V2_k,'fro')+norm(V3_k,'fro');
dual=dmv+dv;
da=frob(A_k)+norm(B1_k,'fro')+norm(B2_k,'fro')+norm(B3_k,'fro');

if  prim<err1+err2*max(pu,dv) && dual<err1+err2*da ; 
    if printyes;
        fprintf('iter = %d');
    end
    break;
end


if k==kmax
    fprintf('iteration reaches maximum.');
end
b=['hatM_k',num2str(k)];
    eval([b,'=vec2tens(vhatM_k,[d1,d2,d3])'])

U1_0 = U1_k;
U2_0 = U2_k;
U3_0 = U3_k;
V1_0 = V1_k;
V2_0 = V2_k;
V3_0 = V3_k;
A_0  = A_k;
B1_0 = B1_k;
B2_0 = B2_k;
B3_0 = B3_k;
hatM_0 = hatM_k;
P_0=  P_k;
b=['U1_k',num2str(k)];
    eval([b,'=U1_0'])
b=['U2_k',num2str(k)];
    eval([b,'=U2_0'])
b=['U3_k',num2str(k)];
    eval([b,'=U3_0'])    
    
 end
 time=toc;
out.time_test = toc;
out.U1_k = U1_k;
out.U2_k = U2_k;
out.U3_k = U3_k;
out.V1_k = V1_k;
out.V2_k=  V2_k;
out.V3_k = V3_k;
out.A_k=  A_k;
out.B1_k = B1_k;
out.B2_k = B2_k;
out.B3_k = B3_k;
out.hatM_k = hatM_k;
out.P_k=  P_k;
out.iter_ada  = k;
   
 U={out.U1_k,out.U2_k,out.U3_k};

 views={M1,N1,L1};
 n_samples = size(views{1},1);  
 epsilon=0.1;
 variances = cell(size(views));
   for i=1:3
        variances{i} =  (double(views{i})'*double(views{i}))/n_samples;
        variances{i} = variances{i} +  epsilon*ones(size(variances{i}));
   end

 Z = zeros(n_samples,d*2);                                  
    for i=1:3
       Z(:,(1+(i-1)*d):(i*d)) = double(views{i})*(pinv(variances{i})^1/2)*U{i};   %z�����帮�1锛�d锛�d+1,锛?2d锛?2d+1锛?3d
    end
  
save stccadata1.txt -ascii Z     %淇�瀛�z

testRatio=0.3;
train_num = 8153;
test_num = 3494;
cca=load("/Users/jianqinsun/Downloads/result_data/ccadata1.txt");

SX100=load("/Users/jianqinsun/Downloads/result_data/sccadata1.txt");
%划分训练集与测试集

% % SX测试数据占全部数据的比例
testRatio = 0.3;
% % 训练集索引
ccatrainIndices = crossvalind('HoldOut', size(cca, 1), testRatio);   % 训练集索引
ccatestIndices = ~ccatrainIndices;                                   % 测试集索引
ccatrainData = cca(ccatrainIndices, :);
ccatrainLabel = label(ccatrainIndices, :);
ccatestData = cca(ccatestIndices, :);
ccatestLabel = label(ccatestIndices, :);
knn_model = fitcknn(ccatrainData,ccatrainLabel(1:train_num),'NumNeighbors',7);
%knn_model = fitcknn(train,train_label(1:train_num),'NumNeighbors',7);
result = predict(knn_model,ccatestData);
acc = 0.;
for i = 1:test_num
    if result(i)==ccatestLabel(i)
        acc = acc+1;
    end
end
fprintf('精确度为：%5.2f%%\n',(acc/test_num)*100);

% -----------------------------------------
% mbcca
mbcca=M1;
mbccatrainIndices = crossvalind('HoldOut', size(mbcca, 1), testRatio);
mbccatestIndices = ~mbccatrainIndices;
mbccatrainData = mbcca(mbccatrainIndices, :);
mbccatrainLabel = label(mbccatrainIndices, :);
mbccatestData = mbcca(mbccatestIndices, :);
mbccatestLabel = label(mbccatestIndices, :);

knn_model = fitcknn(mbccatrainData,mbccatrainLabel(1:train_num),'NumNeighbors',7);
%knn_model = fitcknn(train,train_label(1:train_num),'NumNeighbors',7);
result = predict(knn_model,mbccatestData);
acc = 0.;
for i = 1:test_num
    if result(i)==mbccatestLabel(i)
        acc = acc+1;
    end
end
fprintf('精确度为：%5.2f%%\n',(acc/test_num)*100);



% SX测试数据占全部数据的比例
testRatio = 0.3;
SX100trainIndices = crossvalind('HoldOut', size(SX100, 1), testRatio);
SX100testIndices = ~SX100trainIndices;
SX100trainData = SX100(SX100trainIndices, :);
SXtrainLabel = label(SX100trainIndices, :);
SX100testData = SX100(SX100testIndices, :);
SX100testLabel =label(SX100testIndices, :);

knn_model = fitcknn(SX100trainData,SXtrainLabel(1:train_num),'NumNeighbors',7);
%knn_model = fitcknn(train,train_label(1:train_num),'NumNeighbors',7);
result = predict(knn_model,SX100testData);
acc = 0.;
for i = 1:test_num
    if result(i)==SX100testLabel(i)
        acc = acc+1;
    end
end
fprintf('精确度为：%5.2f%%\n',(acc/test_num)*100);

TX100=load("/Users/jianqinsun/Downloads/result_data/tccadata1.txt");
TTX100=load("/Users/jianqinsun/Downloads/result_data/tccadata2.txt");
TX100trainIndices = crossvalind('HoldOut', size(TX100, 1), testRatio);
TX100testIndices = ~TX100trainIndices;
TX100trainData = TX100(TX100trainIndices, :);
TXtrainLabel = label(TX100trainIndices, :);
TX100testData = TX100(TX100testIndices, :);
TX100testLabel = label(TX100testIndices, :);
knn_model = fitcknn(TX100trainData,TXtrainLabel(1:train_num),'NumNeighbors',7);
%knn_model = fitcknn(train,train_label(1:train_num),'NumNeighbors',7);
result = predict(knn_model,TX100testData);
acc = 0.;
for i = 1:test_num
    if result(i)==TX100testLabel(i)
        acc = acc+1;
    end
end
fprintf('绮剧‘搴�涓猴�?%5.2f%%\n',(acc/test_num)*100);


% TTX

TTX100trainIndices = crossvalind('HoldOut', size(TTX100, 1), testRatio);
TTX100testIndices = ~TTX100trainIndices;
TTX100trainData = TTX100(TTX100trainIndices, :);
TTXtrainLabel = label(TTX100trainIndices, :);
TTX100testData = TTX100(TTX100testIndices, :);
TTX100testLabel = label(TTX100testIndices, :);
knn_model = fitcknn(TTX100trainData,TTXtrainLabel(1:train_num),'NumNeighbors',7);
%knn_model = fitcknn(train,train_label(1:train_num),'NumNeighbors',7);
result = predict(knn_model,TTX100testData);
acc = 0.;
for i = 1:test_num
    if result(i)==TTX100testLabel(i)
        acc = acc+1;
    end
end
fprintf('绮剧‘搴�涓猴�?%5.2f%%\n',(acc/test_num)*100);

STX100=load("/Users/jianqinsun/Downloads/result_data/stccadata1.txt");
STX100trainIndices = crossvalind('HoldOut', size(STX100, 1), testRatio);
STX100testIndices = ~STX100trainIndices;
STX100trainData = STX100(STX100trainIndices, :);
STXtrainLabel = label(STX100trainIndices, :);
STX100testData = STX100(STX100testIndices, :);
STX100testLabel = label(STX100testIndices, :);
knn_model = fitcknn(STX100trainData,STXtrainLabel(1:train_num),'NumNeighbors',7);
%knn_model = fitcknn(train,train_label(1:train_num),'NumNeighbors',7);
result = predict(knn_model,STX100testData);
acc = 0.;

for i = 1:test_num
    if result(i)==STX100testLabel(i)
        acc = acc+1;
        
    end
end

fprintf('绮剧‘搴�涓猴�?%5.2f%%\n',(acc/test_num)*100);