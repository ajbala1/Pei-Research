function synthetic()

testnum=100;
hs=zeros(testnum,1);
ps=zeros(testnum,1);
tstats=zeros(testnum,1);

for i=1:testnum
    fprintf("hello")
    i
% %     %v1
%     x=sin(0:0.5:50);
%     x=x+rand(size(x))*0.1;
%     y=rand(size(x))*10;
    %v2
    x=sin(0:0.5:50);
    x=x+rand(size(x))*0.1;
    y=zeros(size(x));
    for t=1:length(x)
        y(t)=x(t)^3+x(t)^2-x(t)+t*0.02;
    end
    
    alpha=0.05;
    [h,p,tstat]=testPI(x,y,alpha);
    
    if (h>0)&&(tstat<0)
        hs(i)=1;
    else
        hs(i)=0;
    end
    ps(i)=p;
    tstats(i)=tstat;

end

save syntheticv1.mat hs ps tstats 

subplot(1,2,1)
histogram(hs);
xlabel('Hypothesis')
ylabel('Test result (p<0.05)')
subplot(1,2,2);
histogram(tstats)
xlabel('t statistics')
ylabel('Distribution')