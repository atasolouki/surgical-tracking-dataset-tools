function Dataset = packDataset(channel_data, Probe, scanSequence, Phantom, ch_data_time, fs, c, f0)
%packDataset aggregates channel data and required metadata 
%  DATASET = packDataset(CH_DATA, PROBE, SCAN_SEQ, PHANTOM, CHDATA_TIME, FS, C, F0).
%  Inputs: CH_DATA channel data in the format [nRanges, nAzChannels, nElChannels, nTxAz, nTxEl, nPackets, nFrames]
%          PROBE created by Probe()
%          SCAN_SEQ created by scanSequencer()
%          PHANTOM CREATED BY generatePhantom()
%          CHDATA_TIME vector with time samples [s]
%          FS acquisition sampling frequency [Hz]
%          C speed of sound [m/s]
%          F0 pulse centre frequency [Hz]
%
%  Returns: DATASET.ChData channel data
%           DATASET.meta structure with "DataDimensions", "fs", "speedOfSound", "probe", "Sequence" and "Phantom" informtion
%
% Pedro Santos (KU Leuven, 2016)
%
% See also PROBE SCANSEQUENCER GENERATEPHANTOM


% -------------------------------------------------------------------------
% TODO: Bandpassfilter should have a property to define ON/OFF, BW, FC, etc...
%       Also, it doesn't belong to packDataset, but rather to a Processing class
% TODO: to make function and cleanup once objects are contained
% TODO: expand for multiple types of data: ChData, BfData, BmodeData
% -------------------------------------------------------------------------

% Bandpass filter is now done inside readChannelDataHDPULSE
% if 0 && ~isempty(channel_data)% Bandpass filter
% %     filter_params = struct('fs', fs, 'f0', f0, 'bw', 80, 'filter_type', 'fir1');
%     filter_params = struct('fs', fs, 'f0', f0, 'bw', 60, 'filter_type', 'gaussian', 'npadd', 450, 'ncrop', 0, 'tappering', 0.02);
%     channel_data = bandpassFilter(channel_data, filter_params);
%     ch_data_time = ch_data_time(filter_params.ncrop+1:end);
% end


% Store channel data
Dataset.ChData = channel_data;


% Dimensions
meta_description = {'Range', 'Channels Az', 'Channels El', 'Tx Beams Az', 'Tx Beams El', 'Packets', 'Frames'};
for var_i = 1:numel(meta_description)
    Dataset.meta.DataDimensions{var_i,1} = sprintf('%g: %s [%g]', var_i, meta_description{var_i}, size(Dataset.ChData, var_i));
end
Dataset.meta.fs           = fs;
Dataset.meta.speedOfSound = c;


% Probe
Dataset.meta.Probe = Probe;


% Sequence
if size(channel_data,1) > 1 && numel(ch_data_time) ~= size(channel_data,1)
    Utils.error_warning_msg('The number of axial samples in ''channel_data'' does not math the number of timpe points in ''ch_data_time''!', 'fatal', 'Inconsistent dataset');
end    
Dataset.meta.Sequence = scanSequence;
Dataset.meta.Sequence.f0           = f0;
Dataset.meta.Sequence.chSampleTime = ch_data_time;


% Phantom
if ~isempty(Phantom)
    Dataset.meta.Phantom = Phantom;
end



