

function [feat_f,outmeta]=gfeat(f,options,precomp)
feat=getoptions(options,'feat','raw');
outmeta={};
switch feat
  case 'raw'
    feat_f=reshape(f,1,numel(f));
    
  case 'scatt'
    options.dyadic_padding=0;
    options.J=getoptions(options,'J',5);
    feat_f=scatt(f,options);
    
  case 'deloc_scatt'
    options.dyadic_padding=0;
    sf=scatt(f,options);
    feat_f=squeeze(mean(mean(sf,1),2));
    
  case 'comb_deloc_scatt' %score of ESANN paper
    feat_f=combined_scatt(f,options);
    
  case 'comb_deloc_fouriermod'
    feat_f=combined_scatt(f,options);
    
  case 'rilpq' %% use code from http://www.cse.oulu.fi/Downloads/
    feat_f=ri_lpq(f);
    
  case 'lbphf_8_1_16_2_24_3' %% use code from http://www.cse.oulu.fi/Downloads/
    feat_f=lbphf_8_1_16_2_24_3(f,precomp);
    
  case 'kmean_scatt'
    options.dsp=getoptions(options,'dsp',0);
    if (options.dsp)
      if istraining
        st='train';
      else
        st='test';
      end
      
    end
    out=kmeanscatt(f,options);
    feat_f=out.centers(:,3:end);
    
    
  case 'kmean_combined_scatt'
    options.J=getoptions(options,'J',5);
    options.L=getoptions(options,'L',8);
    options.dsp=getoptions(options,'dsp',0);
    if (options.dsp)
      if istraining
        st='train';
      else
        st='test';
      end
      
    end
    [out]=kmeanscatt(f,options);
    feat_f=kmean_combined_scatt(out,options);
    
    
    
  case 'combined_kmean_scatt'
    options.dsp=getoptions(options,'dsp',0);
    if (options.dsp)
      if istraining
        st='train';
      else
        st='test';
      end
      
    end
    feat_f = combined_kmean(f,options);
    
  case 'combined_rot'
    feat_f = mean(combined_scatt2(f,options),1);
  case 'combined_scale_rot'
    feat_f = mean(combined_rot_scale(f,options),1);
    
    
  case 'combined_scale_rot2'
    [out,metaf,metarot,meta,nscatt,mscatt]=combined_rot_scale(f,options);
    feat_f=reshape(out,[nscatt,mscatt,size(out,2)]);
    feat_f=feat_f(2:2:end,2:2:end,:);
    feat_f=reshape(feat_f,size(feat_f,1)*size(feat_f,2),size(feat_f,3));
    
  case 'combined_rot_bandlimited'
    feat_f=combined_rot_bandlimited(f,options);
    
  case 'combined_rot_bandlimited_smoothAtEnd'
    feat_f=combined_rot_bandlimited_smoothAtEnd(f,options);
    
    
  case 'combined_rot_scale_bandlimited'
    
  case 'scattls'
    [SJ1,meta]=scattls(f,options);
    outmeta=meta;
    [nscatt,mscatt,npathscatt]=size(SJ1);
    SJ1lines=reshape(SJ1,nscatt*mscatt,npathscatt);
    feat_f=SJ1lines;
    
  case 'meanscattls'
    [SJ1,meta]=scattls(f,options);
    outmeta=meta;
    [nscatt,mscatt,npathscatt]=size(SJ1);
    SJ1lines=reshape(SJ1,nscatt*mscatt,npathscatt);
    % spatially average to get a consistent estimator
    SJ1avg=mean(SJ1lines,1);
    feat_f=SJ1avg;
    
  case 'affinemeanscattls'
    [feat,meta]=affinemeanscattls(f,options);
    feat_f = squeeze(vectorize(feat))';
    outmeta=meta;
    
  case 'scalemeanscattls'
    [feat_f,outmeta]=scalemeanscattls(f,options);
    
  case 'remainingAffineMeanScattls'
    [feat,meta]=affinemeanscattls(f,options);
    feat_f = squeeze(vectorize(feat))';
    outmeta=meta;
  case 'mfs'
    feat_f=mfsALL(255*f,8,30);
    
  case 'scatt4comb'
    
    % load options for combined scattering
    %options = optionsCombined;
    %options.scattBorderMargin = -1; % for esann
    
    % compute spatial scattering
    % takes 30 s on a core2duo 2.4Ghz for a 640x480 image
    [Sf,meta] = scatt4comb(f,options);
    
    % spatial average
    d = size(Sf,3);
    Sfavg = reshape(mean(mean(Sf,1),2),[1,d]);
    outmeta = meta;
    feat_f = Sfavg;
    % compute combined scattering from spatial scattering
    %[SCf,metaC] = combined(Sfavg,meta,options);
    
    
  case 'scatt4combou'
    options = optionsCombinedOuTex10;  
    % compute spatial scattering
    % takes 30 s on a core2duo 2.4Ghz for a 640x480 image
    [Sf,meta] = scatt4comb(f,options);
    % spatial average
    d = size(Sf,3);
    Sfavg = reshape(mean(mean(Sf,1),2),[1,d]);
    outmeta = meta;
    feat_f = Sfavg;
end
end
