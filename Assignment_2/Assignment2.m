clear;
close all;
clear ;
close all;

samples_number = 10;
bits = [1,0,1,0,1,1,1,0,1,0,1,1,1];
bits_number = length(bits);

% Pulse Shape 

[input] = pulse_shape(bits_number,samples_number,bits);

% input = reshape(input.', [], 1);
% plot(input);


% Channel AWGN 
% Generate Noise To Add

E = 1;
snr_range = -10:1:20;
snr = 10 ^(snr_range(30)/10);

[input_with_noise] = add_noise(bits_number,samples_number,input,E,snr);

plot(input_with_noise);
hold on ;

% filters definations
delta_filter = zeros(1,samples_number);
delta_filter(samples_number/2)=1;
t = 0 : 1 : samples_number -1;
tri_filter = (sqrt(3)/samples_number)*t;
matched_filter = ones(1,samples_number);
filter ={matched_filter,delta_filter,tri_filter};

output = {0,0,0};


for k=1 : 3
    output{k} = conv(input_with_noise,filter{k});
end



% show output of each filter
for k=1 : 3
    % show the output
    figure;
    plot(output{k});
    title(sprintf('output from filter %d', k));
    xlabel('time (ms)');
    ylabel('recieve filter output (bit value)');
    hold on ;
end


% sample the output to get stream of bits
for i=0:bits_number-1
    output_1_samples = sample(output{1},bits_number,samples_number); 
    output_2_samples = sample(output{2},bits_number,samples_number); 
    output_3_samples = sample(output{3},bits_number,samples_number); 
end

% disp(output_1_samples)
% disp(output_2_samples)
% disp(output_3_samples)

% calculate accuracy of each filter 
err_prob_1 = sum(output_1_samples ~= bits);
BER_1 = err_prob_1/bits_number;
err_prob_2 = sum(output_2_samples ~= bits);
BER_2 = err_prob_2/bits_number;
err_prob_3 = sum(output_3_samples ~= bits);
BER_3 = err_prob_3/bits_number;

% disp(BER_1);
% disp(BER_2);
% disp(BER_3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate BER for different SNR 
bits_number = 100000;
samples_number = 10;
indices = randperm(bits_number, 1);
bits= ones(bits_number,1);
bits(indices)=0;

input =pulse_shape(bits_number,samples_number,bits);
snr_range = -10:1:20;
% Preallocate arrays to store BER simulations
BER_sim_1 = zeros (length(snr_range),1);
BER_sim_2 = zeros (length(snr_range),1);
BER_sim_3 = zeros (length(snr_range),1);
BER_theo_1 = zeros(length(snr_range),1);
BER_theo_2 = zeros(length(snr_range),1);
BER_theo_3 = zeros(length(snr_range),1);

for i = 1:length(snr_range)
    snr = 10 ^(snr_range(i)/10);

    input_with_noise = add_noise(bits_number,samples_number,input,E,snr);

    for k = 1:3
        output{k} = conv(input_with_noise, filter{k});  % Consider using 'same' to maintain dimensionality
    end

    % Extracting the middle point for each bit period after convolution
    output_1_samples = sample(output{1},bits_number,samples_number);
    output_2_samples = sample(output{2},bits_number,samples_number);
    output_3_samples = sample(output{3},bits_number,samples_number);

    % disp(size(bits));
    % Calculate errors and BER for each filter
    err_prob_1 = sum(output_1_samples.' ~= bits);
    BER_sim_1(i) = err_prob_1 / bits_number;
    err_prob_2 = sum(output_2_samples.' ~= bits);
    BER_sim_2(i) = err_prob_2 / bits_number;
    err_prob_3 = sum(output_3_samples.' ~= bits);
    disp(BER_sim_1)
    BER_sim_3(i) = err_prob_3 / bits_number;
    BER_theo_1(i)=0.5*erfc(sqrt(snr));
    BER_theo_2(i)=0.5*erfc(sqrt(snr));
    BER_theo_3(i)=0.5*erfc((sqrt(3)/(2)*sqrt(snr)));
end

% Update plot commands to reflect all data
figure;
semilogy(snr_range, BER_theo_1, 'b-');
hold on;
semilogy(snr_range, BER_sim_1, 'm--');
hold off;
title('Matched Filter');
ylim([10^-4 0.5]);
xlabel('SNR (dB)');
ylabel('BER (Log)');
legend('Theoretical', 'Simulation');

figure;
semilogy(snr_range, BER_theo_2, 'r:');
hold on;
semilogy(snr_range, BER_sim_2, 'g--');
hold off;
title('No Filter');
ylim([10^-4 0.5]);
xlabel('SNR (dB)');
ylabel('BER (Log)');
legend('Theoretical', 'Simulation');

figure;
semilogy(snr_range, BER_theo_3, 'c--');
hold on;
semilogy(snr_range, BER_sim_3, 'y--');
hold off;

title('impulse response Filter h(t)=sqrt(3)*t');
ylim([10^-4 0.5]);
xlabel('SNR (dB)');
ylabel('BER (Log)');
legend('Theoretical', 'Simulation');

function [input] = pulse_shape(bits_number,samples_number,bits)
   input = ones(bits_number,samples_number);
    for i=1 : bits_number
        if bits(i) == 0 
        input(i,:) = -input(i,:);
        end
    end
end

function [input_with_noise] = add_noise(bits_number,samples_number,input,E,snr)
    sigma = sqrt(E/(2.0*snr));

    noise = normrnd(0,sigma,[1,bits_number*samples_number]);

    input_with_noise = input;

    % add noise to input
    for i=1 : bits_number        
        input_with_noise(i,:) = input_with_noise(i,:) + noise((samples_number)*(i-1)+1:(samples_number)*(i));
    end
    input_with_noise = reshape(input_with_noise.', [], 1);
end

function [samples]=sample(output,bits_number,samples_number) 
    samples = ones(1,bits_number);
    for i=0:bits_number-1
        samples(i+1) = (output((samples_number - 1) + samples_number * i+1)) > 0;
    end
end