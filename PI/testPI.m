function [h,p,tstat]=testPI(x,y,alpha)

% x=1:0.1:10;
% % y=sin(x.^2+1);
% y=rand(size(x));

%input x and y
%test x->y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%determine embedding dimension d (lag coordinate) and time delay tau for My
fprintf("hello");
taus=1;
ds=1:10;
T=length(y);
neis=1:10;
e_min=Inf;%minimum average error
for i=1:length(ds)
    d=ds(i);
    for j=1:length(taus)
        tau=taus(j);
        %set up My
        My=zeros(T-(d-1)*tau-1,d);
        Targets_y=zeros(T-(d-1)*tau-1,1);
        for t=1:T-(d-1)*tau-1
            My(t,:)=y(t:tau:(t+(d-1)*tau));
            Targets_y(t)=y(t+(d-1)*tau+1);
        end
        
        for k=1:length(neis)
            nei=neis(k);%number of neighbors in method of analogs
            %predict y
            ey=zeros(T-(d-1)*tau-1,1);
            for t=1:T-(d-1)*tau-1
                Y=My(t,:);
                truth=Targets_y(t);
                My_t=My;
                My_t(t,:)=[];
                Targets_t=Targets_y;
                Targets_t(t,:)=[];
                
                pred=analogs(Y,My_t,Targets_t,nei);
                ey(t)=abs(truth-pred);
            end
%             [d,nei,tau,mean(e)]
            if mean(ey)<e_min%better prediction
                e_min=mean(ey);
%                 %optimal prediction error
%                 prederror=ey;
                %optimal hyperpara
                dy_best=d;
                neiy_best=nei;
                tauy_best=tau;
            end
        end
        
    end
end
% dy_best
% neiy_best
% tauy_best
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%determine embedding dimension d and time delay tau for Mx
taus=1;
ds=1:10;
T=length(x);
neis=1:10;
e_min=Inf;%minimum average error
for i=1:length(ds)
    d=ds(i);
    for j=1:length(taus)
        tau=taus(j);
        %set up Mx
        Mx=zeros(T-(d-1)*tau-1,d);
        Targets_x=zeros(T-(d-1)*tau-1,1);
        for t=1:T-(d-1)*tau-1
            Mx(t,:)=x(t:tau:(t+(d-1)*tau));
            Targets_x(t)=x(t+(d-1)*tau+1);
        end
        
        for k=1:length(neis)
            nei=neis(k);%number of neighbors in method of analogs
            %predict y
            ex=zeros(T-(d-1)*tau-1,1);
            for t=1:T-(d-1)*tau-1
                Y=Mx(t,:);
                truth=Targets_x(t);
                Mx_t=Mx;
                Mx_t(t,:)=[];
                Targets_t=Targets_x;
                Targets_t(t,:)=[];
                pred=analogs(Y,Mx_t,Targets_t,nei);
                ex(t)=abs(truth-pred);
            end
%             [d,nei,tau,mean(e)]
            if mean(ex)<e_min%better prediction
                e_min=mean(ex);
%                 %optimal prediction error
%                 prederror=ex;
                %optimal hyperpara
                dx_best=d;
                neix_best=nei;
                taux_best=tau;
            end
        end
        
    end
end
% dx_best
% neix_best
% taux_best


%%%%%%%%%%%%%%%Predict y without x
T=length(y);
tau=tauy_best;
d=dy_best;
nei=neiy_best;

%set up My
My=zeros(T-(d-1)*tau-1,d);
Targets_y=zeros(T-(d-1)*tau-1,1);
for t=1:T-(d-1)*tau-1
    My(t,:)=y(t:tau:(t+(d-1)*tau));
    Targets_y(t)=y(t+(d-1)*tau+1);
end

%predict y
ey=zeros(T-(d-1)*tau-1,1);
for t=1:T-(d-1)*tau-1
    Y=My(t,:);
    truth=Targets_y(t);
    My_t=My;
    My_t(t,:)=[];
    Targets_t=Targets_y;
    Targets_t(t,:)=[];
    pred=analogs(Y,My_t,Targets_t,nei);
    ey(t)=abs(truth-pred);
end

%%%%%%%%%%%%%%%%%%%%%%%Predict y with x
T=length(y);
tau=tauy_best;
d=dy_best;
nei=neiy_best;

%set up Mxy
Mxy=zeros(T-(d-1)*tau-1,d+dx_best);%number of lag coordinate x embedding dimension (dx_best is best embedding dimension for x)
Targets_y=zeros(T-(d-1)*tau-1,1);
for t=1:T-(d-1)*tau-1
    tcnt=t+(d-1)*tau;
    txidx=tcnt:-taux_best:(tcnt-(dx_best-1)*taux_best);
    txidx=txidx(end:-1:1);
    txidx(txidx<1)=1;
    Mxy(t,:)=[y(t:tau:tcnt),x(txidx)];
    Targets_y(t)=y(t+(d-1)*tau+1);
end

%predict y
exy=zeros(T-(d-1)*tau-1,1);
for t=1:T-(d-1)*tau-1
    Y=Mxy(t,:);
    truth=Targets_y(t);
    Mxy_t=Mxy;
    Mxy_t(t,:)=[];
    Targets_t=Targets_y;
    Targets_t(t,:)=[];
    pred=analogs(Y,Mxy_t,Targets_t,nei);
    exy(t)=abs(truth-pred);
end

%%%%%%%%%%%%%%%%% test
% [h,p,ci,stats] = ttest2(exy,ey,'Alpha',0.05);
[h,p,ci,stats] = ttest(exy,ey,'Alpha',alpha);
tstat=stats.tstat;
%null hypothesis: x – y comes from a normal distribution with mean equal 
%to zero and unknown variance, using the paired-sample t-test
%h=0: does not reject with 5% significance level; h=1: reject
%ci: Confidence interval for the difference in population means of x and y,
%default: 95% CI
%stats: tstat>0, exy has larger errors; tstat<0, exy has smaller errors


function pred=analogs(Y,My_t,Targets_t,nei)
%disp(Y)
%disp(My_t)
dist=My_t-ones(size(My_t,1),1)*Y; %size(My_t,1) = # of rows in Mt_y, 
dist=sum((dist.^2),2).^0.5;

%disp(dist)
dist(:,2)=Targets_t;
%disp(dist)
dist=sortrows(dist,1);%sort according to distance, ascending
%disp(dist)
weight=dist(1:nei,1);
disp(weight)
weight=exp(-weight)/sum(exp(-weight));

sample=dist(1:nei,2);
pred=weight'*sample;

