% tutorial for scatternet
%% by Hanlin GU, HKUST
%% Sep 12, 2018
 
% code website httpswww.di.ens.frdatasoftware
% Download and unzip ScatNet latest version-2013 for example
% add path
addpath 'pathtoscatnet';
addpath_scatnet;

% % % %load image a cut out of raphel pictures
% % % filename = 'example.TIF';
% % % x = imreadBW(filename);
% % % imagesc(x);
% % % colormap gray;


imgs=load('SAMPLE/IMG_unknown.mat');
num=size(imgs.Nx_test,3); 
%%%featureNNN=zeros(num,391);

for nnn=1:num
%x=im2double(imgs(:,:,nnn));
x=imgs.Nx_test(:,:,nnn);
fprintf('%d\n',nnn);
%parameter option
filt_opt.J = 5;%5;  
filt_opt.L = 6;%6;  %% not too small   % I find size(feature)=(2*J^2+1)*L
scat_opt.oversampling = 2;
scat_opt.M = 2;  %%too large will consume much time
% filt_opt contains filters-related options
% filt_opt.J (default 4) is the number of scale (j) in the filter bank (psi_{j,theta}). Note that increasing J will also increase the range of translation invariance.
% filt_opt.L (default 8) is the number of orientations. Note that increasing L will also increase the angular selectivity of filters.
% filt_opt.Q (default 1) is the number of scale per octave. Not that increasing Q will decrease the range of translation invariance and increase the scale selectivity of filters.
% scat_opt contains scattering-related options
% scat_opt.M (default 2) is the maximum scattering order. The scat function will compute scattering coefficient of order 0 to M, that is [S_0 x, dotsc, S_M x ]
% scat_opt.oversampling (default 1)  scattering will be oversampled by up to a power of 2.

% Before computing the scattering coefficients of x, 
% we need to precompute the wavelet transform operators 
% that will be applied to the image. Those operators are 
% built by specific built-in factories adapted to different 
% types of signal. For translation-invariant representation 
% of images, we use wavelet_factory_2d.
[Wop, filters] = wavelet_factory_2d(size(x), filt_opt, scat_opt); % Wop is oprator function, handle layer by layer

%Now we call the scat function to compute the scattering of x using those Wop operators.
Sx = scat(x, Wop); % for layers 1, signal has 30; layers 2, signal has 391

% signal is a cell array of 2d matrices, which combines the scattering coefficients, 
% which is indexed by the integer variable (p) as follow
m =2;
p =20;
Sx{m+1}.signal{p};  %p th in m+1 th layer
% meta is a struct with the sequences of (j) and (theta) that corresponds to each (p).
% the sequence of j for p-th coefficient at the m-th order  
Sx{m+1}.meta.j(:,p);
% the sequence of theta for p-th coefficient at the m-th order  
Sx{m+1}.meta.theta(:,p);

%display all filters
display_filter_bank_2d(filters);
colormap gray;


%The variable S is a cell of structure. It is not suited for direct numeric 
%manipulation. You can reformat it in a 3D array using format_scat 
S_mat = format_scat(Sx); % 39160 80 3D matrix



%display all scat representation
shape = size(S_mat);
% for p = 1:shape(1)
%     S_flag=reshape(S_mat(p,:,:),shape(2),shape(3));
%     
% %   %imagesc(reshape(S_mat(p,:,:),shape(2),shape(3)));
% %   imagesc(S_flag);
% %   colormap gray;
% %   %pause(0.001);
%   
% %  dlmwrite('RESULTS/RAF2/S_mat.csv',reshape(S_flag',1,shape(2)*shape(3)),'delimiter',',','-append');
%   
% 
% end


%%%feature extraction
feature = sum(sum(S_mat,2),3);   %%sum every little graph
%featureNNN(nnn,:)=feature';
featureNNN=feature';
dlmwrite('RESULTS/feature_unknown.csv',featureNNN,'delimiter',',','-append');
end

%dlmwrite('RESULTS/label.csv',labs,'delimiter',',','-append');

%%% then you can take classfication like logistic regression, SVM, LDA...