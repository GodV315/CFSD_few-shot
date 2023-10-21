function [y,aa] = modul_SSB(x,Fc,Fs,LSB_or_USB)
    % SSBµ÷ÖÆ
    t = 0:1/Fs:(length(x)-1)/Fs;
    x_hilbert = hilbert(x);
    switch LSB_or_USB
        case 'LSB'
            y = x.*cos(2*pi*Fc*t)+imag(x_hilbert).*sin(2*pi*Fc*t);
        case 'USB'
            y = x.*cos(2*pi*Fc*t)-imag(x_hilbert).*sin(2*pi*Fc*t);
    end
end