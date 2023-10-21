function [y,aa] = modul(x,Fc,Fs)
    % 调制
%     h = cos(2*pi*Fc*(0:1/Fs:(length(x)-1)/Fs));
    h = exp(j*2*pi*Fc*(0:1/Fs:(length(x)-1)/Fs));  %将载波定义为复信号可以避免频谱对称
    y = x.*h;
end