%%在tcca——o运行后加载m张量及u矩阵，对其进行稀疏，输入d1、d2、d3、d4及lambda等参数;

%input
%经过预处理后的各个视角的数据，为x1，x2，x3，x4
%输入各参数，包括各个视角的特征数d及最后提取到子空间的d与lambda的大小


%%%%%%%%

%example
%tccaos(d1,d2,d3,d4,d,kmax,lambda1,lambda2,lambda3,lambda4,rho,tau);

function [Z,f] = tccaos(x1,x2,x3,x4,d1,d2,d3,d4,d,kmax,lambda1,lambda2,lambda3,lambda4,rho,tau)
    
load('M.mat')
load('U1.mat')
load('U2.mat')
load('U3.mat')
load('U4.mat')

A_0=tensor(ones(d1,d2,d3,d4));
B1_0 =zeros(d1,d);  
B2_0 =zeros(d2,d);
B3_0 =zeros(d3,d);
B4_0 =zeros(d4,d);
P_0=tensor(zeros(d,d,d,d));
U1_0 =U1;
U2_0 =U2;
U3_0 =U3;
U4_0 =U4;
V1_0 =U1;
V2_0 =U2;
V3_0 =U3;
V4_0 =U4;
tic;
%%    循环结构 
 for k =1:kmax
%%    P_k  的迭代步
P_k=ttm(M,{U1_0,U2_0,U3_0,U4_0},[1,2,3,4]);

%%    Vp_k 的迭代步
V_11=kron(V4_0,V3_0,V2_0);
V_22=kron(V4_0,V3_0,V1_0);
V_33=kron(V4_0,V2_0,V1_0);
V_44=kron(V3_0,V2_0,V1_0);

tg1=-B1_0+rho*(V1_0-U1_0);
t1=tenmat(P_k,1);
t2=tenmat(P_k,2);
t3=tenmat(P_k,3);
t4=tenmat(P_k,4);
m1=tenmat(M,1);
m2=tenmat(M,2);
m3=tenmat(M,3);
m4=tenmat(M,4);
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

tg4=-B4_0+rho*(V4_0-U4_0);
tff4=-2*tenmat(M,4)*V_44*t4'+V4_0*t4*V_44.'*V_44*t4';
tf4=double(tff4);
F4=tg4+tf4;
%prox
%写成函数形式
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
    
         V4_k = zeros (d3,d);
    for m = 1:2
        if norm(F4(:,m)) >= 1/lambda1
           V4_k(:,m) = V4_0(:,m) - lambda1*(V4_0(:,m)/norm(V4_0(:,m)));
        else
           V4_k(:,m) = 0;
        end
    end
    
    
%%    Up_k 的迭代步
Q_1=double((tenmat(A_0,1)+rho*tenmat(M,1))*V_11*t1')+rho*(V1_0-B1_0);
[U,S,V]=svd(Q_1);
U=U(:,1:d);
U1_k=U*V.';
Q_2=double((tenmat(A_0,2)+rho*tenmat(M,2))*V_22*t2')+rho*(V2_0-B2_0);
[U,S,V]=svd(Q_2);
U=U(:,1:d);
U2_k=U*V.';
Q_3=double((tenmat(A_0,3)+rho*tenmat(M,3))*V_33*t3')+rho*(V3_0-B3_0);
[U,S,V]=svd(Q_3);
U=U(:,1:d);
U3_k=U*V.';
Q_4=double((tenmat(A_0,4)+rho*tenmat(M,4))*V_44*t4')+rho*(V4_0-B4_0);
[U,S,V]=svd(Q_4);
U=U(:,1:d);
U4_k=U*V.';

   
 U={U1_k,U2_k,U3_k,U4_k};

 views={x1,x2,x3,x4};
 n_samples = size(views{1},1);  
 epsilon=0.1;
 variances = cell(size(views));
   for i=1:4
        variances{i} =  (double(views{i})'*double(views{i}))/n_samples;
        variances{i} = variances{i} +  epsilon*ones(size(variances{i}));
   end

 Z = zeros(n_samples,d*2);                                  
    for i=1:4
       Z(:,(1+(i-1)*d):(i*d)) = double(views{i})*(pinv(variances{i})^1/2)*U{i};   %z的列数，1：d，d+1,：2d，2d+1：3d
    end
  
save stccadata1.txt -ascii Z     %保存z %保存z
 end
end



