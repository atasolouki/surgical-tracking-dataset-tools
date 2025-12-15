function BfDataset = packBfDataset(ChDataset, bf_data, BfMeta)
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
% TODO: to make function and cleanup once objects are contained
% TODO: expand for multiple types of data: ChData, BfData, BmodeData
% -------------------------------------------------------------------------


% Store beamformed data
BfDataset = bf_data;

% Copy all metadata from ChData to BfData
BfDataset.meta = ChDataset.meta;

% Update Beamforming meta
BfDataset.meta.Beamforming = BfMeta;

% To Do: we should update this if needed
dt = 2*(BfMeta.imgRanges(2) - BfMeta.imgRanges(1))/ChDataset.meta.speedOfSound;
BfDataset.meta.fs = 1/dt; % If we don't beamform on every time sample then the sampling frequency of our beamformed data changes and so bandpass filtering this will be wrong otherwise.

% Update dimensions
all_fields = fieldnames(BfDataset);
all_fields(ismember(all_fields, 'meta')) = [];  % do not include meta
meta_description = {'Range', 'Lines Az', 'Lines El', 'Packets', 'Frames'};
BfDataset.meta.DataDimensions = [];
for var_i = 1:numel(meta_description)
    BfDataset.meta.DataDimensions{var_i,1} = sprintf('%g: %s [%g]', var_i, meta_description{var_i}, size(BfDataset.(all_fields{1}), var_i));
end





