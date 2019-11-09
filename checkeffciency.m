function efficency=checkeffciency(Btr,Bte,ntr,nte)
Ltrain=size(Btr,2);
Ltest=size(Bte,2);
error=zeros(Ltest,1);
result=zeros(Ltest,1);

%This loop find the shortest euclidean distance for each Bte column
%vector to Btr
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
  
    if j==1
         if intex(i)<=range(j)
        result(i)=1;
         end
    end      
    if j~=1
           if intex(i)<=range(j) & intex(i)>=range(j-1)
            result(i)=1;
           end
    end
    
    
    if range(j)==i*(ntr/nte)
       j=j+1;
    end
end

efficency=sum(result)/size(result,1)
end