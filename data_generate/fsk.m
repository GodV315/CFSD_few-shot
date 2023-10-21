function [y] = fsk(num,M,freqsep,nsamp,Fs)
    % 生成fsk信号
    x = randi([0 M-1],1,num);  % 随机生成num行[0,M-1]的整数的num行整数
    y = fskmod(x,M,freqsep,nsamp,Fs);
end