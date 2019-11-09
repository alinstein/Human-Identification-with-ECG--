clc;
clear all;
close all;
%Written by Alinstein Jose, University of Victoria.
%last modified: April 02, 2019.
%Nopatient:             Number of patients
%Nseg:                  Total number of training and testing segments for patients
%ntr:                   Number of train segments for patients
%nte:                   Number of test segments for patients
%K:                     Size of dictonary is d by K
%st1,st2,st3:           Initial random states.
%d:                     Length of window used to create a local segment
% gp: number of samples a local segment differs from the next local segment.


Nopatient=5;
ntr=10;
nte=10;
d=160;
K=2000;
st1=9;
st2=17;
st3=30;
d=160;
gap=1;

% File ind_mit2019 stores the name of all patients files
Nseg = ntr + nte;
Dtr = [];
Dte = [];
load ind_mit2019
InG = ind_mit2019;
L = length(InG);
rand('state',st1)
Lr = randperm(L);
Lr = Lr(1:Nopatient);
store=[];

%Required data patients are extracted from the file names from index
for i = 1:Nopatient
    
    ni = Lr(i);
    fname = sprintf('a%.0f.mat',InG(ni));
    si = load(fname);
    ai = struct2cell(si);
    di = cell2mat(ai);
    di=di(1:500000);
    store(:,i)= di;
end


%Following loop takes Nseg number of segments(for both testing and-
%training) for all the patients
for j=1:Nopatient
        
        Li = length(store(:,j));
        rand('state',st2+j-1)
        r = randperm(Li);
        for i=1:Nseg
            rq=r(i);
            segtemp(i,:)=store(rq:rq+999,j)';
        end
        Trsegments(j,:,:)=segtemp(1:ntr,:);
        Tesegments(j,:,:)=segtemp(ntr+1:Nseg,:);
end

%p stores numbers of windows applied to each segment
p = 1 + floor((1000 - d)/gap);
Ktr = size(Trsegments,2);

%This loop applies window to training segments to obtain local segments 
for k=1:Nopatient
    for j=1:ntr
        for i=1:p
        temp=Trsegments(k,j,(i-1)*gap+1:(i-1)*gap+d);
        temp=reshape(temp,d,1);
        temp=temp/norm(temp);
        TrainLseg(k,(j-1)*p+i,:)=temp;
        TrainLseg4d(k,j,i,:)=temp;
        end
    end
end

%This loop applies window to testing segments to obtain local segments 
for k=1:Nopatient
    for j=1:nte
        for i=1:p
        temp=Tesegments(k,j,(i-1)*gap+1:(i-1)*gap+d);
        temp=reshape(temp,d,1);
        temp=temp/norm(temp);
        TestLseg(k,(j-1)*p+i,:)=temp;
        TestLseg4d(k,j,i,:)=temp;
        end
    end
end


 TestLseg4d=permute(TestLseg4d,[4 3 2 1]);
 TrainLseg4d=permute(TrainLseg4d,[4 3 2 1]);

%create Random dictionary with dimension d and K
randn('state',st3)
D=randn(d,K);
D=orth(D')';

%lamda is chosen as 0.25/sqrt(d)
lamda=0.25/sqrt(d);

Btr = [];

tic
%This block calculates the sparse vector for each local segment using cvx
%for training set
Btr = [];
for i = 1:Nopatient
    for j = 1:ntr
        t = (i-1)*ntr*p+(j-1)*p;
        Atr = [];
        for k = 1:p
            xk = Xtr(:,t+k);
            cvx_begin quiet
            variable a(K,1)
            minimize(0.5*norm(D*a-xk)+lam*norm(a,1))
            cvx_end
            Atr = [Atr a];
            current_state_tr = [i j k]
             
        end
        b = max(abs(Atr)')';
        Btr = [Btr b];
    end
end

%Following calculates absolute value of each sparse vector
for k=1:Nopatient
    for j=1:ntr
        for i=1:p
        mspar(:,i,j,k)=abs(spar(:,i,j,k));
        end
    end
end

%This block calculates the max pooling, which takes the largest 
%coefficient from all columns 
for k=1:Nopatient
    for j=1:ntr
        beta(:,j,k)=max(mspar(:,:,j,k),[],2);
    end
end


Bte = [];
%This block calculates the sparse vector for each local segment using cvx
%for testing set
tic
for k=1:Nopatient
    for j = 1:nte%Nseg
        Ate = [];
        for i=1:p
            cvx_begin quiet
            variable alp(K,1)
            minimize (  (0.5*norm(TestLseg4d(:,i,j,k)-D*alp)+lamda*norm(alp,1)) )
            cvx_end
            spartes(:,i,j,k)=alp;
            Ate = [Ate alp];
            current_state_tr = [i j k];
        end
          b = max(abs(Ate)')';
          Bte = [Bte b];
    end
end
%end

%Following calculates absolute value of each sparse vector
toc
for k=1:Nopatient
    for j=1:nte
        for i=1:size(TestLseg4d,2)
        mspartes(:,i,j,k)=abs(spartes(:,i,j,k));
        end
    end
end

%This block calculates the max pooling, which takes the largest 
%coefficient from all columns 
for k=1:Nopatient
    for j=1:nte
        betates(:,j,k)=max(mspartes(:,:,j,k),[],2);
    end
end
effciency_of_using_cvx_is = [ checkeffciency(Btr,Bte,ntr,nte)]


tic
%This block calculates the sparse vector for each local segment using
%proximal method for training set
Btr = [];

for k=1:Nopatient
    for j = 1:ntr
        
        Atr = [];
        for i=1:p
        theta2=zeros(2000,1);
        for kq=1:1000
              z1=(theta2+D'*(TrainLseg4d(:,i,j,k)-D*theta2));
             theta2=sign(z1).*max((abs(z1)-lamda/0.9),0);
        end
        alp=theta2;
        spar(:,i,j,k)=alp;
        current_state_tr = [i j k]
        Atr = [Atr alp];
        end
        b = max(abs(Atr)')';
        Btr = [Btr b];
      end
end


%Following calculates absolute value of each sparse vector
for k=1:Nopatient
    for j=1:ntr
        for i=1:p
        mspar(:,i,j,k)=abs(spar(:,i,j,k));
        end
    end
end

%This block calculates the max pooling, which takes the largest 
%coefficient from all columns 
for k=1:Nopatient
    for j=1:ntr
        beta(:,j,k)=max(mspar(:,:,j,k),[],2);
    end
end


Bte = [];
%This block calculates the sparse vector for each local segment using
%proximal method for testing set

for k=1:Nopatient
    for j = 1:nte%Nseg
        Ate = [];
        for i=1:p

            theta2=zeros(2000,1);
            for kq=1:1000
                  z1=(theta2+D'*(TestLseg4d(:,i,j,k)-D*theta2));
                 theta2=sign(z1).*max((abs(z1)-lamda/.9),0);
            end
            alp=theta2;
            spartes(:,i,j,k)=alp;
            Ate = [Ate alp];
            current_state_te = [i j k ]
        end
          b = max(abs(Ate)')';
          Bte = [Bte b];
    end
end


%Following calculates absolute value of each sparse vector
for k=1:Nopatient
    for j=1:nte
        for i=1:size(TestLseg4d,2)
        mspartes(:,i,j,k)=abs(spartes(:,i,j,k));
        end
    end
end

%This block calculates the max pooling, which takes the largest 
%coefficient from all columns 
for k=1:Nopatient
    for j=1:nte
        betates(:,j,k)=max(mspartes(:,:,j,k),[],2);
    end
end
effciency_of_using_proximal_method_is = [ checkeffciency(Btr,Bte,ntr,nte)]
toc

tic
%This block calculates the sparse vector for each local segment using
%ADMM for training set
Btr = [];
r4=0;
d4=0;
alppha=lamda;
Inv=(D'*D+alppha*eye(K));
v=inv(Inv);
for k=1:Nopatient
    for j = 1:ntr
        Atr = [];
        for i=1:p

        xk1(:,1)=zeros(2000,1);lamk(:,1)=zeros(2000,1);
        yk1(:,1)=zeros(2000,1);

        v1=D'*TrainLseg4d(:,i,j,k);
        for f=1 :100
            xk1(:,f+1)=v*(v1+alppha*yk1(:,f)-lamk(:,f));
            kk1=xk1(:,f+1)+lamk(:,f)/alppha;
            yk1(:,f+1)=sign(kk1).*max((abs(kk1)-lamda),0);
            lamk(:,f+1)=lamk(:,f)+alppha*(xk1(:,f+1)-yk1(:,f+1));
            r4(f)=norm(xk1(:,f+1)-yk1(:,f+1),2);
            d4(f)=norm(alppha*(yk1(:,f+1)-yk1(:,f)),2);
            if(r4(f)<1e-3)
                if (d4(f)<1e-3)
                    break;
                end
            end
        end
          
        alp=xk1(:,f);
        spar(:,i,j,k)=alp;
        current_state_tr = [i j k f]
        Atr = [Atr alp];
        end
        b = max(abs(Atr)')';
        Btr = [Btr b];
        
    end
end


%Following calculates absolute value of each sparse vector
for k=1:Nopatient
    for j=1:ntr
        for i=1:p
        mspar(:,i,j,k)=abs(spar(:,i,j,k));
        end
    end
end

%This block calculates the max pooling, which takes the largest 
%coefficient from all columns 
for k=1:Nopatient
    for j=1:ntr
        beta(:,j,k)=max(mspar(:,:,j,k),[],2);
    end
end

r4=0;
d4=0;
Bte = [];
%This block calculates the sparse vector for each local segment using
%ADMM for testing set
for k=1:Nopatient
    for j = 1:nte%Nseg
        Ate = [];
        for i=1:p
                      
        xk1(:,1)=zeros(2000,1);lamk(:,1)=zeros(2000,1);
        yk1(:,1)=zeros(2000,1);

        v1=D'*TestLseg4d(:,i,j,k);
        for f=1 :100
               xk1(:,f+1)=v*(v1+alppha*yk1(:,f)-lamk(:,f));
            kk1=xk1(:,f+1)+lamk(:,f)/alppha;
            yk1(:,f+1)=sign(kk1).*max((abs(kk1)-lamda),0);
            lamk(:,f+1)=lamk(:,f)+alppha*(xk1(:,f+1)-yk1(:,f+1));
            r4(f)=norm(xk1(:,f+1)-yk1(:,f+1),2);
            d4(f)=norm(alppha*(yk1(:,f+1)-yk1(:,f)),2);
            if(r4(f)<1e-3)
                if (d4(f)<1e-3)
                    break;
                end
            end
        end
            
        alp=xk1(:,f);
        spartes(:,i,j,k)=alp;
        Ate = [Ate alp];
        current_state_te = [i j k]
        end
      b = max(abs(Ate)')';
      Bte = [Bte b];
    end
end
%end

%Following calculates absolute value of each sparse vector

for k=1:Nopatient
    for j=1:nte
        for i=1:size(TestLseg4d,2)
        mspartes(:,i,j,k)=abs(spartes(:,i,j,k));
        end
    end
end

%This block calculates the max pooling, which takes the largest 
%coefficient from all columns 
for k=1:Nopatient
    for j=1:nte
        betates(:,j,k)=max(mspartes(:,:,j,k),[],2);
    end
end


effciency_of_using_ADMM_is = [ checkeffciency(Btr,Bte,ntr,nte)]

toc


