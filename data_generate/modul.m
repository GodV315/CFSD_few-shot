function [y,aa] = modul(x,Fc,Fs)
    % ����
%     h = cos(2*pi*Fc*(0:1/Fs:(length(x)-1)/Fs));
    h = exp(j*2*pi*Fc*(0:1/Fs:(length(x)-1)/Fs));  %���ز�����Ϊ���źſ��Ա���Ƶ�׶Գ�
    y = x.*h;
end