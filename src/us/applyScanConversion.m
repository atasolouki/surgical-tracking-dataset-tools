function [scanConvData] = applyScanConversion(dataIn, meta, debug_plots)

% TODO: memory allocation

if ~ exist('debug_plots', 'var') || isempty(debug_plots), debug_plots = 0; end

nFrames = size(dataIn, 5);
nPackets = size(dataIn, 4);

% Setup waitbar
if nFrames*nPackets > 1
    h_w = waitbar(0, 'Scan converting data. Please wait...');
    update_val = max(1,round(nFrames*nPackets/20)); % update every 20
    fprintf('\t Computing scan conversion (%gx%g pixels)\t ... \t', meta.PostProcessing.outImgSize);
end

% Loop through all frames and packets
t0 = tic;
for frame_i = 1:nFrames
    for packet_i = 1:nPackets

        if (nFrames*nPackets > 1) && ~rem((frame_i-1)*nPackets+packet_i, update_val)
            try waitbar(((frame_i-1)*nPackets+packet_i)/nFrames*nPackets, h_w); end
        end

        thisScanConv = scanConvertThis(dataIn(:,:,:,packet_i, frame_i), meta, debug_plots);
        %         thisScanConv = scanConvertThis(dataIn(:,:,1,packet_i, frame_i), meta, debug_plots);

        if packet_i == 1 && frame_i == 1
            scanConvData = thisScanConv;

            % Allocate space for other frames
            if nPackets*nFrames > 1
                scanConvData.B_mode(:,:,:,nPackets,nFrames) = zeros(size(thisScanConv.B_mode));
            end
        else
            scanConvData.B_mode(:,:,:,packet_i, frame_i) = thisScanConv.B_mode;
        end
    end
end

try
    waitbar(((frame_i-1)*nPackets+packet_i)/nFrames*nPackets, h_w);
    delete(h_w)
end

if nFrames*nPackets > 1, fprintf('Done! (%.2f sec)\n', toc(t0)); end
end

function [scanConvData] = scanConvertThis(dataIn, meta, debug_plots)

if ~ exist('debug_plots', 'var') || isempty(debug_plots), debug_plots = 0; end

% When no PostProcessing was done, look for imgRanges and bf_angles in Beamforming
try
    bf_ranges = meta.PostProcessing.imgRanges;
    bf_angles = meta.PostProcessing.bf_angles;
    % slope_at_apex = -obj.tx_apex(1) / sqrt(R^2 - obj.tx_apex(1)^2);
    % theta = atand(slope_at_apex); % degree
    % bf_angles = meta.PostProcessing.bf_angles + theta;
catch
    bf_ranges = meta.Beamforming.imgRanges;
    bf_angles = meta.Beamforming.bf_angles;
end

Bmode = single(dataIn);
outImgSize  = meta.PostProcessing.outImgSize; % [dep az]

if isfield(meta.Sequence.txEvents(1) , 'line_pos')
    line_pos = meta.Sequence.txEvents(1).line_pos;
    scanConvData = scanConversion_PWI(Bmode, line_pos, bf_ranges, outImgSize);
else
    if isfield(meta.PostProcessing, 'forceBmodeAz') && ~isempty(meta.PostProcessing.forceBmodeAz)
        % Due to memory limitations, we need to reconstrcuct multiple subsectors in HD-Pulse. If we want to shwo them independently, we need to force Az limits
        bf_angles = [linspace(meta.PostProcessing.forceBmodeAz(1), meta.PostProcessing.forceBmodeAz(2), size(dataIn, 2));...
            zeros(1, size(dataIn, 2))];
    end
    az_min_max = bf_ranges(end)*sind([bf_angles(1, 1) bf_angles(1, end)]);
    el_min_max = bf_ranges(end)*sind([bf_angles(2, 1) bf_angles(2, end)]);

    size_az    = az_min_max(2) - az_min_max(1);
    size_el    = el_min_max(2) - el_min_max(1);
    size_dep   = bf_ranges(end);
    % dt         = mean(diff(bf_ranges)) / meta.speedOfSound *2;

    if ndims(Bmode) == 2
        %     fprintf('\t Computing scan conversion (%gx%g pixels)\t ... \t', outImgSize(1:2))
        scanConvData = scanConversion(Bmode, bf_angles(1,:), bf_ranges, outImgSize(1:2)); % Scan convert

    elseif size(Bmode, 2) == 1 % if single txAz, squeeze dimensions (tmp)
        %     fprintf('\t Computing scan conversion (%gx%g pixels)\t ... \t', outImgSize(1:2))

        Bmode = squeeze(Bmode);
        tmpScanConv = scanConversion(Bmode, bf_angles(2,:), bf_ranges, outImgSize([1 3])); % Scan convert
        tmpScanConv(:,1,:,:,:) = tmpScanConv; % add Az dimention (singleton)
        scanConvData.B_mode   = tmpScanConv;
        scanConvData.disp_el_lims = scanConvData.disp_az_lims; % It was actually elevation angles
        rmfield(scanConvData, 'disp_az_lims');

    elseif ndims(Bmode) == 3
        dep_min = bf_ranges(1);
        dep_max = bf_ranges(end);

        bf_angles_az = unique(bf_angles(1,:))';
        bf_angles_el = unique(bf_angles(2,:))';
        lat_min_az = bf_ranges(end)*sind(bf_angles_az(1));
        lat_max_az = bf_ranges(end)*sind(bf_angles_az(end));
        lat_min_el = bf_ranges(end)*sind(bf_angles_el(1));
        lat_max_el = bf_ranges(end)*sind(bf_angles_el(end));

        pos_vec_x_new = (0:1/(outImgSize(2)-1):1).*(lat_max_az-lat_min_az) + lat_min_az;
        pos_vec_y_new = (0:1/(outImgSize(3)-1):1).*(lat_max_el-lat_min_el) + lat_min_el;
        pos_vec_z_new = (0:1/(outImgSize(1)-1):1).*(dep_max-dep_min) + dep_min;
        [pos_mat_x_new, pos_mat_y_new, pos_mat_z_new] = ndgrid(pos_vec_x_new, pos_vec_y_new, pos_vec_z_new);

        % convert new points to polar coordinates
        [th_cart, r_cart, z_cart] = Utils.cart2beam(pos_mat_x_new, pos_mat_y_new, pos_mat_z_new);

        % interpolate using linear interpolation
        b_mode = interp3(bf_angles_az, bf_ranges, bf_angles_el, Bmode, th_cart, z_cart, r_cart, 'linear');
        b_mode = permute(b_mode,[3 1 2]);

        Bmode_mask = ones(size(b_mode));
        Bmode_mask(isnan(b_mode)) = 0;
        b_mode(isnan(b_mode)) = min(min(b_mode(:)));

        scanConvData.B_mode = b_mode;
        scanConvData.Bmode_mask = Bmode_mask;
        scanConvData.disp_az_lims = pos_vec_x_new;
        scanConvData.disp_el_lims = pos_vec_y_new;
        scanConvData.disp_dep_lims = pos_vec_z_new;

    else
        %     fprintf('\t Computing scan conversion from COLE (%gx%gx%g pixels)\t ... \t', outImgSize)

        sys.start_depth     = bf_ranges(1);
        sys.end_depth       = bf_ranges(end);
        sys.width           = bf_angles_az(end)*2/180*pi;
        sys.width_elev      = bf_angles_el(end)*2/180*pi;
        sys.res_axial       = numel(bf_ranges);
        sys.nr_lines        = size(Bmode, 2);
        sys.nr_lines_elev   = size(Bmode, 3);

        FOI                 = [sys.start_depth sys.end_depth az_min_max(1) az_min_max(2) el_min_max(1) el_min_max(2)];
        imageSize           = outImgSize;

        scanConvData.B_mode = ScanConvertCOLE(permute(Bmode, [2 1 3]), sys, FOI, imageSize);
        scanConvData.B_mode = permute(scanConvData.B_mode, [2 1 3]);
        scanConvData.disp_az_lims  = (0:1/(imageSize(2)-1):1).*size_az - size_az/2;
        scanConvData.disp_el_lims  = (0:1/(imageSize(3)-1):1).*size_el - size_el/2;
        scanConvData.disp_dep_lims = (0:1/(imageSize(1)-1):1).*size_dep;
    end
end

if debug_plots
    x_pos = 1000*scanConvData.disp_az_lims;
    y_pos = 1000*scanConvData.disp_dep_lims;
    figure, imagesc(x_pos, y_pos, scanConvData.B_mode(:,:,round(end/2)))
    title('Scan Converted'), xlabel('Azimuth [mm]'), ylabel('Range [mm]'), axis equal tight, grid on, colorbar
end
end