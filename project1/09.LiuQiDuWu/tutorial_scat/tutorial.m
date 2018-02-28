% tutorial for image processing using 
% scattering toolbox 
% author : laurent sifre laurent.sifre@polytechnique.edu

%%
clear;
addpath('common')
addpath('combined')
addpath('display')
addpath('2d')
%%
%load MNIST images
x = load('./mnist-matlab/mnist.mat');
x_train_6000 = x.training.images(:,:,1:6000);
x_test_1000 = x.test.images(:,:,1:1000);
x_train_6000_labels = x.training.labels(1:6000, 1);
x_test_1000_labels = x.test.labels(1:1000, 1);

tic();
features = zeros(1000, 241);
for i = 1:1000
    [Sx,meta] = scatt(x_test_1000(:,:,i));
    features(i, :) = meta.scatt_ave;
    if mod(i, 100) == 0
        display(i);
    end
end


toc();
%%
% display all paths successively
for p = 1:size(Sx,3)
  imagesc(Sx(:,:,p));
  pause(0.1);
end

%% 
% look at the meta 
subplot(311);
plot(meta.order);
subplot(312);
plot(meta.scale);
subplot(313);
plot(meta.orientation);

%%
% compute scattering with non-default parameters
options.J = 5;
options.L = 8;
[Sx,meta] = scatt(x,options);

%%
% look at the filters
options.J = 5;
options.L = 6;
filters = gabor_filter_bank_2d([128,128],options);

resolution = 1;
j = 4;
theta = 3;
filt = ifft2(filters.psi{resolution}{j}{theta});
clf;
imagesc([real(fftshift(filt)),imag(fftshift(filt))]);
%%
% look at all the filter at once
imagesc(display_filter_spatial_all(filters,32));


%%
% scattering display
clear options;
% retrieve some additional meta information
options.renorm_study = 1;
[Sx,meta] = scatt(x,options);
disp  = fulldisplay2d(squeeze(Sx(16,16,:)),meta);
figure(1);
imagesc(disp{1});
figure(2);
imagesc(disp{2});