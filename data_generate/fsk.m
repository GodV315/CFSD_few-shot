function [y] = fsk(num,M,freqsep,nsamp,Fs)
    % ����fsk�ź�
    x = randi([0 M-1],1,num);  % �������num��[0,M-1]��������num������
    y = fskmod(x,M,freqsep,nsamp,Fs);
end