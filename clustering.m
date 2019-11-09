
% ntr=10;nte=10;
Ltrain=size(Btr,2);
Ltest=size(Bte,2);
error=zeros(Ltest,1);
result=zeros(Ltest,1);
for i=1:Ltest
   for j=1:Ltrain
      error(j)=norm(Btr(:,j)-Bte(:,i)); 
   end
    [M,I] = min(error);
     intex(i)=I;
end
range=1:ntr;
range=ntr:ntr:Ltrain;
j=1;
for i=1:Ltest
  
    if intex(i)<=range(j)
        result(i)=1;
    end
    
    if range(j)==i*(ntr/nte)
       j=j+1;
    end
end

efficency=sum(result)/size(result,1)
    