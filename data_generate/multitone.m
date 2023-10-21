function y = multitone(time_len,M,freqseq,Fs)
%MULTITONE 此处显示有关此函数的摘要
%   此处显示详细说明
    t = 0:1/Fs:time_len-1/Fs;
    freq = -M/2*freqseq+freqseq/2 + [0:M-1]*freqseq;
    y = 0;
    for i = 1:M
        y = y+exp(1j*2*pi*freq(i)*t);
    end
end

