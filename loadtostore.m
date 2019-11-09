function [s]=loadtostore(st1,Nopatient)

load ind_mit2019
InG = ind_mit2019;
L = length(InG);
rand('state',st1)
Lr = randperm(L);
Lr = Lr(1:Nopatient);
store=[];

for i = 1:Nopatient
    ni = Lr(i);
    fname = sprintf('a%.0f.mat',InG(ni));
    si = load(fname);
    ai = struct2cell(si);
    di = cell2mat(ai);
    di=di(1:500000);
    store(:,i)= di;
end
s=store;
end