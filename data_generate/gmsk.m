function modwave = gmsk(num, Fs, Rs)
    span = 8;  % 符号数
    Len = span + 1 ; % symbol time span after Gaussian pulse shaping
    BT = 0.5;  % 时间积
    msg = randi([0,1], 1, num);
    sps = round(Fs / Rs);    
    %% 高斯滤波
    csps = 200 * sps ; % high sample rate for discrete integral
    % 升采样
    gaussfilter = gaussdesign(BT, span, csps) ; 
    rectbasewav = rectpulse([1], csps) ; % 阶跃函数
    gt = conv(gaussfilter, rectbasewav) ;% 得到gt
    deltaphi = zeros(Len, sps) ;
    for i = 1:Len  % 一个信号可以影响到四个码元，长度为4*2+1 = 9
        for j = 1:sps       
            deltaphi(Len+1-i, j) = sum(gt(1+(i-1)*csps : 1+(i-1)*csps+j*csps/sps-1)) / sum(gt) * pi/2;                
        end 
    end  % 求出 gt的积分，长度为 9 T    
    %% MSK调制
    bitmsg = 2*msg - 1 ; % 转成双极性
    padbimsg = [zeros(1, (Len-1)/2), bitmsg, zeros(1, (Len-1)/2)] ;  % 前后增加4个长度
    phitrace = zeros(1,length(msg)) ;
    lastphi = 0 ; 
    for k = 1:length(msg)
        kthphi = lastphi + padbimsg(k : k+Len-1) * deltaphi;
        lastphi = kthphi(end) ;
        phitrace = [phitrace kthphi] ; 
    end
    t = (0:length(msg)*sps-1)/Fs ;
    modwave = cos(phitrace) + 1i*sin(phitrace); 
    modwave = modwave(length(msg)+1:end);
end

