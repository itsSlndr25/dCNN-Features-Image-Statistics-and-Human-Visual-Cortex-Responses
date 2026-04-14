function [slope, sse]=getImageSpectSlope(inputImage)


[ny, nx]=size(inputImage); % 425x425

ftImg=fft2(inputImage); % could use rfft2 in py

if(mod(ny,2))
    fymap=[0:(ny-1)/2,(ny-1)/2:-1:1]'; % (0~212,212~1)
else
    fymap=[0:(ny)/2-1,(ny)/2:-1:1]';
end    
fymap=fymap*ones(1,nx); 

if(mod(nx,2))
    fxmap=[0:(nx-1)/2,(nx-1)/2:-1:1];
else
    fxmap=[0:(nx)/2-1,(nx)/2:-1:1];
end    
fxmap=ones(ny,1)*fxmap;

rmap=sqrt(fxmap.^2+fymap.^2); % creat a map indicate distance to edge

maxR=floor(max(max(rmap))); % pick the max value of rmap

% amp=NaN(maxR);
% amps=NaN(maxR);
% freq=NaN(maxR);
counter=0;
    for i = 2:ceil(0.8*maxR) % 2:maxR | 2:ceil(0.8*maxR)
       findex=find(rmap>i-0.5 & rmap<i+0.5); % index of certain radius i+-0.5
        if(length(findex)>0)
           counter=counter+1;
           temp=abs(ftImg(findex)); % amplitude at the index radius 
           amp(counter)=mean(temp);
           amps(counter)=std(temp);
           freq(counter)=i;

        end
    end

    p=polyfit(log(freq),log(amp),1); % log-log plot (on both freq & amplitude
    predy=polyval(p,log(freq)); % evaluate polynomial p at each point of log(freq)
    
    sse=sum((predy-log(amp)).^2);
    slope=p(1);
   plot(log(freq),log(amp),'o',log(freq),(predy),'-')
   hold on
   plot(log(freq),log(amp-amps),':',log(freq),log(amp+amps),':')
   hold off
end    


