function y = psk(num,M,Rs,Fs,aa)
    %Éú³ÉpskÐÅºÅ
    x = randi([0 M-1],1,num);
%     x_upsample = kron(x,ones(1,nsamp));
    msgmod_1 = pskmod(x,M,pi/M);   
    fs = round(Fs/Rs);
    y = rcosflt(msgmod_1,1,fs,'fir/sqrt',aa).'; 
%     y = msgmod_1;
    L = (length(y)-fs*length(msgmod_1))/2;
    y = y(L+1:end-L);
end


