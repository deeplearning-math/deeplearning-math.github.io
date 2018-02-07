% Combined scattering for OUTEX 10 databases
%
% ---------!!!!!!!!!!!!!!!!!!-----------
% ---------BEFORE YOU BEGIN :-----------
% ---------!!!!!!!!!!!!!!!!!!-----------
% add this to your startup.m file 
% --------------------------------------
% addpath '/pathtocombinedscatteringtoolbox/'
% addpath '/pathtocombinedscatteringtoolbox/scattering/common/'
% addpath '/pathtocombinedscatteringtoolbox/scattering/1d/'
% addpath '/pathtocombinedscatteringtoolbox/scattering/2d/'
% addpath '/pathtocombinedscatteringtoolbox/misc/'
% global mpath
% mpath='/pathtooutexdatabase/';
% ---------------------------------------

startup;
% load database
Ntrain = 20;
Ntest = 160;
Nclass = 24;

Ntrain = 2;
Ntest = 2;
Nclass = 2;
% about 800M ram
[train,test] = retrieve_ou_tex_10(Ntrain,Ntest,Nclass);

%%
% configure combined scattering
options.feat = 'scatt4combou';
scattering = @(x)(gfeat(x,options));
[~,meta] = scattering(train{1}{1});
options.Jc = 3;
combinedScattering = @(x) (combined(scattering(x),meta,options));

%%
% compute all scattering vectors
% about 7 hours on a 2,4 ghz core 2 duo
for c = 1:Nclass
  fprintf('\n train class %d : ',c);
  for i=1:Ntrain
    fprintf('%d..',i);
    if (mod(i,10)==0) fprintf('\n'); end
    scatteringVectors{c}{i} = combinedScattering(train{c}{i});
  end
  fprintf('\n test class %d : ',c);
  for i=1:Ntest
    fprintf('%d..',i);
    if (mod(i,10)==0) fprintf('\n'); end
    scatteringVectors{c}{i+Ntrain} = combinedScattering(test{c}{i});
  end
end
%%
% classify

options.classif = 'kNN';
options.split = 1;
options.Ncut = Ntrain;
results = generic_classifier_laurent(scatteringVectors,{},options);
