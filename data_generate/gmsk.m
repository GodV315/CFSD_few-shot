function modwave = gmsk(num, Fs, Rs)
    span = 8;  % ������
    Len = span + 1 ; % symbol time span after Gaussian pulse shaping
    BT = 0.5;  % ʱ���
    msg = randi([0,1], 1, num);
    sps = round(Fs / Rs);    
    %% ��˹�˲�
    csps = 200 * sps ; % high sample rate for discrete integral
    % ������
    gaussfilter = gaussdesign(BT, span, csps) ; 
    rectbasewav = rectpulse([1], csps) ; % ��Ծ����
    gt = conv(gaussfilter, rectbasewav) ;% �õ�gt
    deltaphi = zeros(Len, sps) ;
    for i = 1:Len  % һ���źſ���Ӱ�쵽�ĸ���Ԫ������Ϊ4*2+1 = 9
        for j = 1:sps       
            deltaphi(Len+1-i, j) = sum(gt(1+(i-1)*csps : 1+(i-1)*csps+j*csps/sps-1)) / sum(gt) * pi/2;                
        end 
    end  % ��� gt�Ļ��֣�����Ϊ 9 T    
    %% MSK����
    bitmsg = 2*msg - 1 ; % ת��˫����
    padbimsg = [zeros(1, (Len-1)/2), bitmsg, zeros(1, (Len-1)/2)] ;  % ǰ������4������
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

