close all;clear;clc

Fs = 50e3;      % Sample rate (Hz)
total_time = 5;  % �ź��ܳ�5��
total_len = Fs*total_time;
f_seq_ini = [1e3, 10e3];  % �ź�֮��Ƶ�ʼ��(���ϸ��źŵ�frequ���¸��źŵ�freqd)
start_int = [0, 3] * Fs; % �ź����λ�ã�0s��0~3s��Ӧ��λ�ã�
BW_len_int = [2, 5] * Fs; % �źų���
BW_len_seq_ini = [0.1, 0.3] * Fs;  % ͻ�����
SNR_dB_ini = [10, 20];
signal_types = {'2FSK','4FSK','8FSK','GMSK', 'PSK', 'Morse', 'AM-DSB', 'AM-USB', 'AM-LSB', 'FM', 'LFM', '8-Tone', '16-Tone'};
signal_types_iwant = {'2FSK','4FSK','PSK', 'Morse', 'AM-DSB', 'FM','GMSK','8-Tone', '16-Tone'};
channel_types = {'None','iturHFLQ','iturHFLM','iturHFLD','iturHFMQ','iturHFMM','iturHFMD','iturHFMDV','iturHFHQ','iturHFHM','iturHFHD'};
dir_background = 'backgrounds';
NumPerBackground = 2;
output_dir = '..\data\label_2FSK_4FSK_PSK_Morse_AM-DSB_FM_GMSK_8-Tone_16-Tone\few-shot_data';
% �����ȡһ��background���ü���[5s, 125kHz]
path = dir([dir_background, '\*.wav']);
filelist = {path.name};
randIndex = randperm(size(filelist,2));
filelist = filelist(randIndex);
% ���ֵ�·��
path = dir('songs\*.mp3');
songlist = {path.name};
for idx_background = 2: size(filelist, 2)
    background_name = filelist{idx_background};
    audiopath = fullfile(dir_background, background_name);
    [y_ini, fs] = audioread(audiopath);  % y��������˫ͨ��
    y_ini = (y_ini(:,1) + 1i*y_ini(:,2) )';    
    fix_len = 5 * Fs;
    if (length(y_ini) < fix_len) || (fs < Fs)
        print([audiopath, '�ߴ��С']);
        exit();
    else
        % �ü�(ֻ����DDC��ʱ��ü����������ʱ����Ϊ5s)
        if fs > Fs
            dfc = fs/2-(fs-Fs)/2+randi(fs-Fs);
            y_DDC = y_ini .* exp(-1i*2*pi*dfc*(1:length(y_ini))/fs);
            y_DDC_LPF = lowpass(y_DDC, Fs/2, fs);
            y_DDC_LPF_Dsample = resample(y_DDC_LPF,Fs/1e3, round(fs/1e3));
        elseif fs == Fs
            y_DDC_LPF_Dsample = y_ini;
        end        
        start = randi(length(y_DDC_LPF_Dsample)-fix_len+1);
        y_DDC_LPF_Dsample_clip = y_DDC_LPF_Dsample(start: start+fix_len-1);
    end

    for idx_NumPerBackground = 1: NumPerBackground
        y = zeros(size(y_DDC_LPF_Dsample_clip));
        % ���ɱ�ע�ı�
        filedir = fullfile(output_dir,[strrep(background_name,'.wav',''),'_',...
        num2str(idx_NumPerBackground)]);
        if ~exist([filedir, '.wav'])                    
            % һ�������и��źŵı�ע
            freqU = [0];
            freqD = [0];
            stime = [0];
            etime = [0];
            content = {0};
            while 1
                type = signal_types_iwant{randi(length(signal_types_iwant))};                 
                SNR_dB = randi([SNR_dB_ini(1), SNR_dB_ini(2)]);                
                switch type
                    case {'2FSK', '4FSK', '8FSK'}
                        % һ·�źŵĲ���������ͻ����ͬ�Ĳ�����
                        Rs = randi([50, 100]);  % ����
                        M = str2double(type(1));  % MFSK                        
                        freqseq = randi([200,300]);  % FSKƵ�ʼ��
                        nsamp = round(Fs/Rs);  % ÿ�ַ�������Ŀ
                        B = (M-1)*freqseq + 2*Fs/nsamp;  % ����
                        f_seq = randi(f_seq_ini);
                        Fc = freqU(end) + f_seq + B/2;
                        if Fc + B/2 >= Fs  % �˳������źŵ��ж�
                            break;
                        end
                        frequ = ceil(Fc + B/2);
                        freqd = floor(Fc - B/2);                     
                        BW_len = BW_len_int(1) + randi(BW_len_int(2)-BW_len_int(1));  % ͻ���ĳ���                
                        num = round(BW_len/nsamp);  % ͻ���������ַ���Ŀ              
                        k = fsk(num,M,freqseq,nsamp,Fs);  %fskmod�����Ǹ��ź�
                        BW_len = length(k);
                        if BW_len>total_len
                            k = k(1:total_len);
                            BW_len = total_len;
                        end
                        start = randi(total_len-BW_len+1)-1;
                        stop = start + BW_len;   
                        k_modul = modul(k,Fc,Fs);
                        k_modul_scale = scale_by_snr(k_modul,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB);  % ����SNR�ͱ�����Ӧ��������������źŷ���,ֱ�Ӽ���total_len���ȵı����������������ź�û�и��ǵ��������е�ʱ�̶���ɷ��ȹ���
                        y(start+1: stop) = y(start+1: stop) + k_modul_scale(1: BW_len);
                        freqU = [freqU, frequ];
                        freqD = [freqD, freqd];
                        stime = [stime, start/Fs];
                        etime = [etime, stop/Fs];
                        content{end+1} = [num2str(M), 'FSK'];

                    case 'PSK'
                        % һ·�źŵĲ���������ͻ����ͬ�Ĳ�����
                        Rs = randi(4) * 2400;  % ���٣�ʹ�������12.8kHz
                        M = 2^(randi([1,3]));  % MFSK
                        aa = 0.25;  % �����˲���ϵ��
                        B = 2 * ceil((1-2*aa)/(1-aa)*Rs);  % ����
                        f_seq = randi(f_seq_ini);
                        Fc = freqU(end) + f_seq + B/2;
                        if Fc + B/2 >= Fs  % �˳������źŵ��ж�
                            break;
                        end
                        frequ = ceil(Fc + B/2);
                        freqd = floor(Fc - B/2);                                                         
                        BW_len = BW_len_int(1) + randi(BW_len_int(2)-BW_len_int(1));  % ͻ���ĳ���                
                        num = round(BW_len/Fs*Rs);  % ͻ���������ַ���Ŀ              
                        k = psk(num,M,Rs,Fs,aa);  % pskmod�����Ǹ��ź�
                        BW_len = length(k);
                        if BW_len>total_len
                            k = k(1:total_len);
                            BW_len = total_len;
                        end
                        start = randi(total_len-BW_len+1)-1;
                        stop = start + BW_len;  
                        k_modul = modul(k,Fc,Fs);                        
                        k_modul_scale = scale_by_snr(k_modul,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB);                            
                        y(start+1: stop) = y(start+1: stop) + k_modul_scale(1: BW_len);
                        freqU = [freqU, frequ];
                        freqD = [freqD, freqd];
                        stime = [stime, start/Fs];   
                        etime = [etime, stop/Fs];
                        content{end+1} = ['PSK'];                                                   

                    case 'GMSK' 
                        % һ·�źŵĲ���������ͻ����ͬ�Ĳ�����
                        Rs = randi([1e2, 1e3]);  % ����
                        nsamp = round(Fs / Rs);
                        num_freq = 2;  % MFSK                        
                        B = 3*Rs/2;  % ����(*3���˹�������Ϊ�˿�����̫խ)
                        f_seq = randi(f_seq_ini);
                        Fc = freqU(end) + randi(f_seq_ini) + B/2;                        
                        if Fc + B/2 >= Fs  % �˳������źŵ��ж�
                            break;
                        end   
                        frequ = ceil(Fc + B/2);
                        freqd = floor(Fc - B/2);                                               
                        BW_len = BW_len_int(1) + randi(BW_len_int(2)-BW_len_int(1));  % ͻ���ĳ���
                        num = round(BW_len / Fs * Rs);
                        k = gmsk(num, Fs, Rs);
                        BW_len = length(k);
                        if BW_len>total_len
                            k = k(1:total_len);
                            BW_len = total_len;
                        end
                        start = randi(total_len-BW_len+1)-1;
                        stop = start + BW_len;
                        k_modul = modul(k,Fc,Fs);
                        k_modul_scale = scale_by_snr(k_modul,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB);                              
                        y(start+1: stop) = y(start+1: stop) + k_modul_scale(1: BW_len);
                        freqU = [freqU, frequ];
                        freqD = [freqD, freqd];  
                        stime = [stime, start/Fs];
                        etime = [etime, stop/Fs];                        
                        content{end+1} = 'GMSK';                                                     

                    case 'Morse'
                        time_dit = 0.1+(0.2-0.1)*rand(1,1);  % dit����Ϊ0.1~0.2��
                        B = [100, 70];  % Morse�Ŀ�����Ϊ�̶�100+70Hz
                        f_seq = randi(f_seq_ini);
                        Fc = freqU(end) + f_seq + B(1);                        
                        if Fc + B(1) >= Fs  % �˳������źŵ��ж�
                            break;
                        end   
                        frequ = ceil(Fc + B(1));
                        freqd = floor(Fc - B(2));
                        S = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
                        k = randi([1 26],1,randi([2 5]));
                        text = S(k);
                        morsecode = morse(Fs,time_dit,Fc,text,0);
                        if length(morsecode) > total_len  % �ض�ʱ�䳤��     
                            morsecode = morsecode(1:total_len);
                        end
                        BW_len = length(morsecode);
                        start = randi(total_len-length(morsecode)+1)-1;
                        stop = start + BW_len;
                        morsecode_scale = scale_by_snr(morsecode,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB);                       
                        y(start+1: stop) = y(start+1: stop) + morsecode_scale;
                        freqU = [freqU, frequ];
                        freqD = [freqD, freqd];
                        stime = [stime, start/Fs];
                        etime = [etime, stop/Fs];
                        content{end+1} = ['Morse']; 

                    case 'AM-DSB'
                        k = randi([1 length(songlist)]);
                        audiopath = cell2mat(fullfile('songs',songlist(k)));                        
                        [audio,Fs_audio] = audioread(audiopath);                        
                        if length(audio(:,1)) > Fs_audio * total_time  % �ض�audio�����ٺ��������
                            m = randi([30*Fs_audio length(audio)-30*Fs_audio-Fs_audio * total_time]);  % ���ֵ�ǰ��30����ԣ����ⳤʱ�������
                            audio = audio(m: m+Fs_audio*total_time-1, :);
                        end
                        audio = audio(:,1)';  % ֻȡһ��ͨ�����źţ�Ϊʵ�ź�
                        B = randi([3,12]) * 1e3;  % ����Ϊ3~12kHz
                        f_seq = randi(f_seq_ini);
                        Fc = freqU(end) + f_seq + B/2;                        
                        if Fc + B/2 >= Fs  % �˳������źŵ��ж�
                            break;
                        end  
                        frequ = ceil(Fc + B/2);
                        freqd = floor(Fc - B/2);
                        f_cut = B / 2 - 1e3;  % LPF��ֹƵ��(-1e3����Ϊlowpass�صò��ɾ�)
                        audio_LPF = lowpass(audio, f_cut, Fs_audio, 'Steepness', 0.95);            
                        audio_resample = resample(audio_LPF, Fs, Fs_audio);                       
                        audio_resample_modul = ammod(audio_resample,Fs/4,Fs);                                                
                        audio_resample_modul = hilbert(audio_resample_modul);
                        audio_resample_modul = modul(audio_resample_modul,Fc-Fs/4,Fs);
                        audio_resample_modul = audio_resample_modul(1:randi([ceil(0.5*total_len), total_len]));  % ����ü�
                        BW_len = length(audio_resample_modul);
                        start = randi(total_len-length(audio_resample_modul)+1)-1;
                        stop = start + BW_len;
                        audio_resample_modul_scale = scale_by_snr(audio_resample_modul,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB);     
                        y(start+1: stop) = y(start+1: stop) + audio_resample_modul_scale;            
                        freqU = [freqU, frequ];
                        freqD = [freqD, freqd];
                        stime = [stime, start/Fs];
                        etime = [etime, stop/Fs];
                        content{end+1} = ['AM-DSB'];

                    case 'AM-USB'
                        k = randi([1 length(songlist)]);
                        audiopath = cell2mat(fullfile('songs',songlist(k)));
                        [audio,Fs_audio] = audioread(audiopath);
                        if length(audio(:,1)) > Fs_audio * total_time  % �ض�audio�����ٺ��������
                            m = randi([30*Fs_audio length(audio)-30*Fs_audio-Fs_audio * total_time]);  % ���ֵ�ǰ��30����ԣ����ⳤʱ�������
                            audio = audio(m: m+Fs_audio*total_time-1, :);
                        end
                        audio = audio(:,1)'; 
                        B = randi([2,6]) * 1e3;  % ����Ϊ2~6kHz
                        f_seq = randi(f_seq_ini);
                        Fc = freqU(end) + f_seq;
                        if Fc + B >= Fs  % �˳������źŵ��ж�
                            break;
                        end
                        frequ = ceil(Fc+B);
                        freqd = Fc;
                        f_cut = B-1e3;  % LPF��ֹƵ��(-1e3����Ϊlowpass�صò��ɾ�)
                        audio_LPF = lowpass(audio, f_cut, Fs_audio, 'Steepness', 0.95);            
                        audio_resample = resample(audio_LPF, Fs, Fs_audio);    
                        audio_resample_modul = modul_SSB(audio_resample,B,Fs,'USB');  % �ȵ�����BHz���Ȼ�õ��ߴ������ϱ�Ƶ����Ҫ��Ƶ����ȥ��
                        audio_resample_modul = hilbert(audio_resample_modul);  % hilbert�任ȥ���ԳƲ���
                        audio_resample_modul = modul(audio_resample_modul,Fc-B,Fs);  
                        audio_resample_modul = audio_resample_modul(1:randi([ceil(0.5*total_len), total_len]));  % ����ü�
                        BW_len = length(audio_resample_modul);
                        start = randi(total_len-length(audio_resample_modul)+1)-1;
                        stop = start + BW_len;
                        audio_resample_modul_scale = scale_by_snr(audio_resample_modul,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB);  
                        y(start+1: stop) = y(start+1: stop) + audio_resample_modul_scale;            
                        freqU = [freqU, frequ];
                        freqD = [freqD, freqd];
                        stime = [stime, start/Fs];
                        etime = [etime, stop/Fs];
                        content{end+1} = ['AM-USB'];

                    case 'AM-LSB'
                        k = randi([1 length(songlist)]);
                        audiopath = cell2mat(fullfile('songs',songlist(k)));
                        [audio,Fs_audio] = audioread(audiopath);
                        if length(audio(:,1)) > Fs_audio * total_time  % �ض�audio�����ٺ��������
                            m = randi([30*Fs_audio length(audio)-30*Fs_audio-Fs_audio * total_time]);  % ���ֵ�ǰ��30����ԣ����ⳤʱ�������
                            audio = audio(m: m+Fs_audio*total_time-1, :);
                        end
                        audio = audio(:,1)'; 
                        B = randi([2,6]) * 1e3;  % ����Ϊ2~6kHz
                        f_seq = randi(f_seq_ini);
                        Fc = freqU(end) + f_seq+B;
                        if Fc>= Fs  % �˳������źŵ��ж�
                            break;
                        end
                        frequ = Fc;
                        freqd = floor(Fc-B);
                        f_cut = B;  % LPF��ֹƵ��(-1e3����Ϊlowpass�صò��ɾ�)
                        audio_LPF = lowpass(audio, f_cut, Fs_audio, 'Steepness', 0.95);            
                        audio_resample = resample(audio_LPF, Fs, Fs_audio);    
                        audio_resample_modul = modul_SSB(audio_resample,B,Fs,'LSB');  % �ȵ�����BHz���Ȼ�õ��ߴ������ϱ�Ƶ����Ҫ��Ƶ����ȥ��
                        audio_resample_modul = hilbert(audio_resample_modul);  % hilbert�任ȥ���ԳƲ���
                        audio_resample_modul = modul(audio_resample_modul,Fc-B,Fs);  
                        audio_resample_modul = audio_resample_modul(1:randi([ceil(0.5*total_len), total_len]));  % ����ü�
                        BW_len = length(audio_resample_modul);
                        start = randi(total_len-length(audio_resample_modul)+1)-1;
                        stop = start + BW_len;
                        audio_resample_modul_scale = scale_by_snr(audio_resample_modul,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB);  
                        y(start+1: stop) = y(start+1: stop) + audio_resample_modul_scale;            
                        freqU = [freqU, frequ];
                        freqD = [freqD, freqd];
                        stime = [stime, start/Fs];
                        etime = [etime, stop/Fs];
                        content{end+1} = ['AM-LSB'];

                    case 'FM'
                        B = randi([3,10])*1e3; 
                        k = randi([1 length(songlist)]);
                        audiopath = cell2mat(fullfile('songs',songlist(k)));
                        [audio,Fs_audio] = audioread(audiopath);
                        if length(audio(:,1)) > Fs_audio * total_time  % �ض�audio�����ٺ��������
                            m = randi([30*Fs_audio length(audio)-30*Fs_audio-Fs_audio * total_time]);  % ���ֵ�ǰ��30����ԣ����ⳤʱ�������
                            audio = audio(m: m+Fs_audio*total_time-1, :);
                        end
                        audio = audio(:,1).';  % �����ź�ȡʵ�ź�
                        f_cut = randi([3,6])*0.5*1e3-1e3;  % LPF�ض�Ƶ��                       
                        freq_dev = B/2; % FM�����Ǹ�ֵ������
                        Fc = freqU(end) + randi(f_seq_ini) + B/2;
                        if Fc + B/2 >= Fs  % �˳������źŵ��ж�
                            break;
                        end
                        frequ = ceil(Fc + B/2);
                        freqd = floor(Fc - B/2);                                 
                        audio_resample = resample(audio, Fs, Fs_audio);      
                        audio_resample_modul = fmmod(audio_resample,Fs/4,Fs,freq_dev);  % ��Ϊaudio_resample��ʵ�źţ�ʵ�ź��ϱ�Ƶ��FcӦ��Fs/2
                        audio_resample_modul = hilbert(audio_resample_modul);  % ʵ�źű临�ź�
                        audio_resample_modul = modul(audio_resample_modul,Fc-Fs/4,Fs);
                        if length(audio_resample_modul) > total_len  % �ض�ʱ�䳤��
                            m = randi([30*Fs_audio length(audio)-30*Fs_audio-Fs_audio * total_time]);  % ���ֵ�ǰ��30����ԣ����ⳤʱ�������
                            audio_resample_modul = audio_resample_modul(m:m+Fs*randi([ceil(0.5*total_len/Fs) total_len/Fs])-1);
                        end
                        BW_len = length(audio_resample_modul);
                        start = randi(total_len-length(audio_resample_modul)+1)-1;
                        stop = start + BW_len;
                        audio_resample_modul_scale = scale_by_snr(audio_resample_modul,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB); 
                        y(start+1: stop) = y(start+1: stop) + audio_resample_modul_scale;            
                        freqU = [freqU, ceil(Fc + B/2)];
                        freqD = [freqD, floor(Fc - B/2)];
                        stime = [stime, start/Fs];
                        etime = [etime, stop/Fs];
                        content{end+1} = ['FM'];

                    case 'LFM'                        
                        B = randi([5,10])*1e3;
                        Fc = freqU(end) + randi(f_seq_ini) + B/2;
                        if Fc + B/2 >= Fs  % �˳������źŵ��ж�
                            break;
                        end
                        frequ = Fc+B/2;
                        freqd = Fc-B/2;
                        time_len = randi([total_time/2*Fs,total_time*Fs])/Fs;
                        t = 0:1/Fs:time_len-1/Fs;
                        T = randi([8,30])*0.001 * (B/1e3);  % ɨƵ�����Ǵ���/1e3��0.008~0.03������б�ʣ�
                        k = chirp(mod(t,T),0,T,B);  % ������0~BƵ���ڵ��źţ��ٵ�����Fc
                        k = hilbert(k);  % ȥ���ԳƲ���
                        k_modul = modul(k,freqd,Fs);
                        if length(k_modul) > total_len
                            k_modul = k_modul(1:total_len);
                        end
                        BW_len = length(k_modul);
                        start = randi(total_len-BW_len+1)-1;
                        stop = start + BW_len;
                        k_modul_scale = scale_by_snr(k_modul,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB); 
                        y(start+1: stop) = y(start+1: stop) + k_modul_scale;            
                        freqU = [freqU, frequ];
                        freqD = [freqD, freqd];
                        stime = [stime, start/Fs];
                        etime = [etime, stop/Fs];
                        content{end+1} = ['LFM'];

                    case {'8-Tone', '16-Tone'}                         
                        num_freq = strsplit(type,'-');                        
                        num_freq = str2double(num_freq{1});
                        Rs = randi([10,20])*10;
                        freqseq = 2.5*Rs;  % Ƶ�ʼ�����ÿһ·�����һ��
                        B = freqseq*(num_freq-1) + 2*Rs;
                        M = 2^(randi([1,3]));  % MPSK
                        aa = 0.25;  % �����˲���ϵ��
                        Fc = freqU(end) + randi(f_seq_ini) + B/2;
                        if Fc + B/2 >= Fs  % �˳������źŵ��ж�
                            break;
                        end
                        frequ = Fc+B/2;
                        freqd = Fc-B/2;
                        time_len = randi([total_time/2*Fs,total_time*Fs])/Fs;
                        Fc_per_freq = Fc - num_freq/2*freqseq + freqseq/2 + [0:num_freq-1]*freqseq;
                        k_modul = 0;
                        for i = 1:length(Fc_per_freq)
                            Fc_freq = Fc_per_freq(i);
                            num = floor(time_len*Rs);
                            k_freq = psk(num,M,Rs,Fs,aa);
                            k_freq_modul = modul(k_freq,Fc_freq,Fs);
                            k_modul = k_modul + k_freq_modul;
                        end
                        if length(k_modul) > total_len
                            k_modul = k_modul(1:total_len);
                        end
                        BW_len = length(k_modul);
                        start = randi(total_len-BW_len+1)-1;
                        stop = start + BW_len;
                        k_modul_scale = scale_by_snr(k_modul,y_DDC_LPF_Dsample_clip,0,total_len,freqd,frequ,Fs,SNR_dB); 
                        y(start+1: stop) = y(start+1: stop) + k_modul_scale;            
                        freqU = [freqU, frequ];
                        freqD = [freqD, freqd];
                        stime = [stime, start/Fs];
                        etime = [etime, stop/Fs];
                        content{end+1} = [num2str(num_freq),'-Tone'];                        
                end    
            end
            % �ź�ͨ������ŵ�            
            channel_quality = channel_types{randi(length(channel_types))};
            if strcmp(channel_quality,'None')             
                y = y;
            elseif strcmp(channel_quality,'AWGN')
                y = awgn(y,randi([10,20]),'measured');
            else                
                fd = 1;   %max freqz shift (Hz)
                chan = stdchan(1/Fs, fd, channel_quality);
                y = filter(chan, y);
            end
            fid = fopen([filedir, '.DR.txt'],'w');
            for i = 2: length(freqU)
                FreqU = freqU(i);
                FreqD = freqD(i);
                DateTimeStart = round(max(stime(i), 0) * 1e7);
                DateTimeEnd = round(min(etime(i), total_time) * 1e7);
                Content = content{i};
                bbox = struct("FreqD",FreqD,"FreqU",FreqU,"DateTimeStart",DateTimeStart,...
                        "DateTimeEnd",DateTimeEnd,"Content",Content);
                bbox_json = jsonencode(bbox);
                fprintf(fid,[bbox_json,'\r\n'],'utf-8');
            end
            % ���ӱ�����������Ƶ
            y = y + y_DDC_LPF_Dsample_clip;
            y = y / max(abs(y));  % �����ݹ�һ��
            y = y.';  % ���������ת�ò����á�'��������Ṳ��
            y_2channel = zeros(length(y), 2);
            y_2channel(:, 1) = real(y);
            y_2channel(:, 2) = imag(y);
            audiowrite([filedir, '.wav'],y_2channel, Fs);
            fclose(fid);
%             spectrogram(y, 4096, 3500, 4096, Fs, 'yaxis');
        end
    end
end