function [PostData] = PostProcessing(BfDataSet, field_name, debug_plots)
%  [DATA]=PostProcessing(BFDATASET, FIELD_NAME, DEBUG_PLOTS)
%  Inputs:  BFDATASET: Contains the beformed data and metadata from offlineBeamforming().
%                      It also includes the meta.PostProcessing struct defining some pos-processing settings.
%           FIELD_NAME:  (optional) string selecting which field in BFDATASET to postprocess (default: 'BfData')
%           DEBUG_PLOTS: (optional) TRUE to plot intermediate processing stages
%
%  Returns: DATA: Is a structure containing metadata and one or more data stages. Selecting the data to return
%           can be one by adding its fieldname in the 'PostSetup.dataToSave' cell flag.
%               DATA.Bmode: (default) scan converted data
%               DATA:meta: (always returned) metadata with updated post-processing values
%           Intermediate processing stages:
%               DATA.RawBf: raw beamformed
%               DATA.Filt: gaussian bandpass filtered
%               DATA.TGC: time-gain compensated
%               DATA.Envelope: envelope detected
%               DATA.IncCompounding: incoherent compounded
%               DATA.Norm: normalized
%               DATA.Log: log compressed
%               DATA.Interp: beam space interpolated
%               DATA.EndProcessed: final processed data before scan conversion
%
%  Vangjush Komini,Pedro Santos (KU Leuven 2016)

% See also PACKDATASET PACKBFDATASET SCANSEQUENCER OFFLINEBEAMFORMING


if ~exist('debug_plots', 'var') || isempty(debug_plots), debug_plots = 0; end
if ~exist('field_name', 'var') || isempty(field_name), field_name = 'BfData'; end

BfData     = BfDataSet.(field_name);
BfSetup    = BfDataSet.meta.Beamforming;
if isfield(BfDataSet.meta, 'PostProcessing')
    PostSetup  = BfDataSet.meta.PostProcessing;
else
    PostSetup = [];
end
if isfield(PostSetup, 'dataToSave')
    dataToSave = PostSetup.dataToSave;
else
    dataToSave = {'scanConv'};   % default
end

fs = BfDataSet.meta.fs;
f0 = BfDataSet.meta.Sequence.f0;
c0 = BfDataSet.meta.speedOfSound;

bf_angles   = BfSetup.bf_angles;
bf_ranges   = BfSetup.imgRanges;
bf_range_dz = bf_ranges(2) - bf_ranges(1);
rx_apex     = BfSetup.rxApex;
n_bf_lines  = BfSetup.nScanLines;
n_bf_ranges = length(bf_ranges);
n_packets   = size(BfData, 4);
n_frames    = size(BfData, 5);

if any(cell2mat(strfind(dataToSave, 'RawBf'))), PostData.RawBf = BfData; end

if debug_plots
    figure, imagesc(bf_angles(1,:), 1000*bf_ranges, BfData(:,:,round(n_bf_lines(2)/2),1,1))
    title('Before Post-Processing'), xlabel('Az angle [deg]'), ylabel('Range [mm]'), grid on; colorbar;
end

%% -----------------Padd near field---------------------------------
range_to_pad = rx_apex(3) : bf_range_dz : bf_ranges(1)-bf_range_dz;
bf_ranges    = [range_to_pad, bf_ranges];       % update range vector
padd         = zeros(length(range_to_pad), n_bf_lines(1), n_bf_lines(2), n_packets, n_frames);
if ~isempty(padd)
    BfData   = [padd; BfData];      % now pad the Bf image
end

% Padd also aftwerwards (to avoid filtering artifacts)
if isfield(PostSetup, 'use_post_pad') && ~isempty(PostSetup.use_post_pad)
    max_depth    = PostSetup.use_post_pad;
    range_to_pad = bf_ranges(end)+bf_range_dz : bf_range_dz : max_depth;
    bf_ranges    = [bf_ranges, range_to_pad];       % update range vector
    padd         = zeros(length(range_to_pad), n_bf_lines(1), n_bf_lines(2), n_packets, n_frames);
    if ~isempty(padd)
        BfData   = [BfData; padd];      % now pad the Bf image
    end
    nPaddingAfter = numel(range_to_pad);
    n_bf_ranges   = length(bf_ranges);
end

%% -----------------Filter---------------------------------
if isreal(BfData) %we don't want to filter the IQ data as it's already filtered during the demodulation process
    if isfield(PostSetup, 'filterBmode') &&  PostSetup.filterBmode == 1
        if ~isfield(PostSetup, 'filt_BW') || isempty(PostSetup.filt_BW)
            PostSetup.filt_BW = 80;
        end
        if ~isfield(PostSetup, 'filt_fc') || isempty(PostSetup.filt_fc)
            PostSetup.filt_fc = f0;
        end

        filter_params = struct('fs', fs, 'f0', PostSetup.filt_fc, 'bw', PostSetup.filt_BW, 'filter_type', 'gaussian');
        BfData = bandpassFilter(BfData, filter_params);

        if any(cell2mat(strfind(dataToSave, 'Filt'))), PostData.Filt = BfData; end

        if debug_plots
            figure, imagesc(bf_angles(1,:), 1000*bf_ranges, BfData(:,:,round(n_bf_lines(2)/2),1,1))
            title('Filtered'), xlabel('Az angle [deg]'), ylabel('Range [mm]'), grid on; colorbar;
        end

    end
end

%% --------------------TGC----------------------------------------------
if ~isfield(PostSetup, 'alpha') || isempty(PostSetup.alpha)
    PostSetup.alpha = 4e-6; % TODO: empirical value -> may need correction
end
tgc = exp(2*PostSetup.alpha*bf_ranges*f0)';
BfData = bsxfun(@times, BfData, tgc);   % store intermediate data

if any(cell2mat(strfind(dataToSave, 'TGC'))), PostData.TGC = BfData; end

if debug_plots
    figure, imagesc(bf_angles(1,:), 1000*bf_ranges, BfData(:,:,round(n_bf_lines(2)/2),1,1))
    title('TGC'), xlabel('Az angle [deg]'), ylabel('Range [mm]'), grid on; colorbar
end

%% ----------------Envelope detection---------------------------------
if isreal(BfData)  % if not IQ data
    BfData = abs(hilbert(reshape(BfData, numel(bf_ranges), [])));
else  % if IQ data
    BfData = abs(reshape(BfData, numel(bf_ranges), []));
end
BfData = reshape(BfData, numel(bf_ranges), n_bf_lines(1), n_bf_lines(2), n_packets, n_frames); % back to [nRanges, nAzLines, nElLines, nPackets, nFrames]

if any(cell2mat(strfind(dataToSave, 'Envelope'))), PostData.Envelope = BfData; end


if debug_plots
    figure, imagesc(bf_angles(1,:), 1000*bf_ranges, BfData(:,:,round(n_bf_lines(2)/2),1,1))
    title('Envelope detection'), xlabel('Az angle [deg]'), ylabel('Range [mm]'), grid on; colorbar;
end

%% -----------------Incoherent compounding---------------------------------
if BfDataSet.meta.Sequence.compoundingMode == 2
    BfData = computeCompounding(BfData, BfDataSet.meta);
end

if any(cell2mat(strfind(dataToSave, 'IncCompounding'))), PostData.IncCompounding = BfData; end

%% ----------------Log compression---------------------------------
if ~isfield(PostSetup, 'norm') || isempty(PostSetup.norm)
    PostSetup.norm = max(BfData(:));
end
BfData = BfData ./ PostSetup.norm;
if any(cell2mat(strfind(dataToSave, 'Norm'))), PostData.Norm = BfData; end


BfData = 20*log10(BfData);
if any(cell2mat(strfind(dataToSave, 'Log'))), PostData.Log = BfData; end


if debug_plots
    figure, imagesc(bf_angles(1,:), 1000*bf_ranges, BfData(:,:,round(n_bf_lines(2)/2),1,1))
    title('Log compressed'), xlabel('Az angle [deg]'), ylabel('Range [mm]'), grid on   colorbar
end

%% ----------------Interpolation Beamspace---------------------------------------
if isfield(PostSetup, 'postBfInterp') && ~isempty(PostSetup.postBfInterp) && ~isequal(PostSetup.postBfInterp, 0)
    if size(BfData,4) * size(BfData, 5) > 1
        % not done for packets/frames
        keyboard
    end

    if ndims(BfData) == 3 && ~any(n_bf_lines == 1) % 3-D data (no sinle-line in any dimension)

        dep_in = 1:size(BfData, 1);
        az_in  = 1:size(BfData, 2);
        el_in  = 1:size(BfData, 3);

        dep_out = linspace(1, dep_in(end), dep_in(end)*PostSetup.postBfInterp(1));
        az_out  = linspace(1, az_in(end), az_in(end)*PostSetup.postBfInterp(2));
        el_out  = linspace(1, el_in(end), el_in(end)*PostSetup.postBfInterp(3));

        BfData = interpn(dep_in, az_in', el_in, BfData, dep_out, az_out', el_out);   % store intermediate data

        % TODO: return new beam coordinates

    elseif n_bf_lines(1) == 1 % Elevation slice

        ranges_out    = linspace(bf_ranges(1), bf_ranges(end), n_bf_ranges*PostSetup.postBfInterp(1));
        el_angles_out = linspace(bf_angles(2,1), bf_angles(2,end), n_bf_lines(2)*PostSetup.postBfInterp(3));

        [dep_in, el_in]   = ndgrid(bf_ranges, bf_angles(2,:));
        [dep_out, el_out] = ndgrid(ranges_out, el_angles_out);

        BfData(:,1,:,:,:) = interpn(dep_in, el_in, squeeze(BfData), dep_out, el_out);   % store intermediate data

        % Update dimensions
        bf_angles   = [0*el_angles_out; el_angles_out];
        bf_ranges   = ranges_out;
        % bf_range_dz = bf_ranges(2) - bf_ranges(1);
        n_bf_lines  = [1 length(el_angles_out)];
        % n_bf_ranges = length(ranges_out);

        if debug_plots
            figure, imagesc(el_angles_out, 1000*bf_ranges, squeeze(BfData(:,round(n_bf_lines(1)/2),:)))
            title('Interpolated'), xlabel('El angle [deg]'), ylabel('Range [mm]'), grid on   colorbar
        end

    else

        ranges_out    = linspace(bf_ranges(1), bf_ranges(end), n_bf_ranges*PostSetup.postBfInterp(1));
        az_angles_out = linspace(bf_angles(1,1), bf_angles(1,end), n_bf_lines(1)*PostSetup.postBfInterp(2));
        %         el_angles_out = linspace(bf_angles(2,1), bf_angles(2,end), n_bf_lines(2)*PostSetup.postBfInterp(3));

        [dep_in, az_in]   = ndgrid(bf_ranges, bf_angles(1,:));
        [dep_out, az_out] = ndgrid(ranges_out, az_angles_out);

        BfData = interpn(dep_in, az_in, BfData, dep_out, az_out);   % store intermediate data

        % Update dimensions
        bf_angles   = [az_angles_out; 0*az_angles_out];
        bf_ranges   = ranges_out;
        % bf_range_dz = bf_ranges(2) - bf_ranges(1);
        n_bf_lines  = [length(az_angles_out) 1];
        % n_bf_ranges = length(ranges_out);

        if debug_plots
            figure, imagesc(az_angles_out, 1000*bf_ranges, BfData(:,:,round(n_bf_lines(2)/2)))
            title('Interpolated'), xlabel('Az angle [deg]'), ylabel('Range [mm]'), grid on   colorbar
        end

    end

    if any(cell2mat(strfind(dataToSave, 'Interp'))), PostData.Interp = BfData; end

end

%% Remove padding now

BfData(1:length(range_to_pad), :, :, :, :) = [];
bf_ranges(1:length(range_to_pad)) = [];
% n_bf_ranges = length(bf_ranges);

if isfield(PostSetup, 'use_post_pad') && ~isempty(nPaddingAfter)
    BfData(end-nPaddingAfter+1:end, :, :, :, :) = [];
    bf_ranges(end-nPaddingAfter+1:end) = [];
    % n_bf_ranges = length(bf_ranges);
end

%% Force maximum depth
if isfield(PostSetup, 'max_Bmode_range') && ~isempty(PostSetup.max_Bmode_range)
    end_dep = PostSetup.max_Bmode_range;
    samples_to_crop = find(bf_ranges > end_dep);
    bf_ranges(samples_to_crop) = [];
    BfData(samples_to_crop, :,:,:,:) = [];
end

if any(cell2mat(strfind(dataToSave, 'Processed_beamSpace'))), PostData.EndProcessed = BfData; end

%% -----------------Scan Conversion---------------------------------
if ~isfield(BfDataSet.meta.PostProcessing, 'outImgSize') || isempty(PostSetup.outImgSize)
    BfDataSet.meta.PostProcessing.outImgSize = [512 512 512]; % default B-mode size in pixels
    PostSetup.outImgSize = [512 512 512];
end
if any(cell2mat(strfind(dataToSave, 'scanConv')))
    if isfield(BfDataSet.meta.PostProcessing, 'ScanConvMode') && strcmp(BfDataSet.meta.PostProcessing.ScanConvMode,'Anatomical') % 2D
        [~,i]=max(bf_angles(1, 2:end)-bf_angles(1, 1:end-1));
        bf_angles = [bf_angles(:,1:i) bf_angles(:,i)*3/2-bf_angles(:,i-1)/2 bf_angles(:,i+1)*3/2-bf_angles(:,i+2)/2 bf_angles(:,i+1:end)]; % padd a line of zeros in between ROIs
        BfData=[BfData(:,1:i) min(BfData(:))*ones(size(BfData,1),2) BfData(:,i+1:end)];
    end

    BfDataSet.meta.PostProcessing.imgRanges = bf_ranges;
    BfDataSet.meta.PostProcessing.bf_angles = bf_angles;
    BfData = applyScanConversion(BfData, BfDataSet.meta, []);

    PostData.Bmode          = BfData.B_mode;
    PostSetup.Bmode_mask    = BfData.Bmode_mask;
    PostSetup.disp_az_lims  = BfData.disp_az_lims;
    try PostSetup.disp_el_lims = BfData.disp_el_lims; end
    PostSetup.disp_dep_lims = BfData.disp_dep_lims;
end

%% Store metadata
PostData.meta = BfDataSet.meta;
PostData.meta.PostProcessing = PostSetup; % update PostProcessing meta

%Created by Bidisha -> need for all the space conversions/speqle/filtering
sys.f = f0;
sys.c = c0;
sys.fs = fs;
sys.spatial_fs = c0/2/fs;
sys.res_axial = PostData.meta.PostProcessing.outImgSize(1);
sys.start_depth = min(bf_ranges(:));
sys.nr_lines = n_bf_lines(1);
sys.axial_samples = length(bf_ranges(:));
sys.end_depth = max(bf_ranges(:));
sys.width =    (max(bf_angles(:)) - min(bf_angles(:)))*pi/180;
sys.angles = (bf_angles/180)*pi;
sys.range = bf_ranges;
PostData.meta.sys = sys;
