function output = scale_by_snr(signal,noise,start,stop,freqd,frequ,Fs,SNR_dB)
%SCALE_BY_SNR �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
% �����źŹ���
energy_signal = sum(abs(signal).^2);
power_signal = energy_signal / length(signal);                           
% �����ź�����ʱ�䡢Ƶ�ʷ�Χ�ڱ����Ĺ���
noise_clip = noise(start+1: stop);
num_fft = length(noise_clip);
noise_clip_fft = fft(noise_clip) / num_fft;
noise_clip_fft = noise_clip_fft(floor(freqd/Fs*num_fft)+1:ceil(frequ/Fs*num_fft));
power_noise_clip = sum(abs(noise_clip_fft).^2);
scale = sqrt(power_noise_clip*10^(SNR_dB/10)/power_signal);
output = scale * signal;

end