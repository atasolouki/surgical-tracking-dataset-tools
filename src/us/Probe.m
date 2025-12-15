%--------------------------------------------------------------------------
% TODO: allow more modes in plotAperture (e.g. element nr or delay)
%--------------------------------------------------------------------------



classdef Probe

    properties (SetAccess = protected)
        name            = '';                       % Probe name
    end


    properties
        nElementsAz                                 % Number of Az elements
        nElementsEl                                 % Number of El elements
        pitchAz                                     % Distance between elements centers [m]
        pitchEl                                     % Distance between elements centers [m]
        kerfAz          = 0;                        % Gap between transducer elements [m]
        kerfEl          = 0;                        % Gap between transducer elements [m]
        ROC             = [0 0];                    % Radius of curvatire in az & el [m]
        rx_probe_origo  = [0 0 0]/1000;             % Origin for the receive array [m]
        tx_probe_origo  = [0 0 0]/1000;             % Origin for the transmit array [m]
        el_focus        = 0;                        % Probe elvation focus (ROC) [m] - Not used
        elemenPos                                   % Element positions [nAz nEl [x|y|z]] [m]
        MuxerPos                                    % Muxer centre positions [nAz nEl [x|y|z]] [m]
        connectorPinout                             % Map of which elements are connected to the ordered channels

        type            = 'xdc_linear_array';       % Transducer type (used for Field II)
        nSubAz          = 5;                        % umber of mathematical elements Field II [az]
        nSubEl          = 11;                       % umber of mathematical elements Field II [el]
        debug_mode      = 0;                        % TRUE to plot stages of probe creation

        usePinout       = 0;                        % TRUE to reshuffle element->channel mapping as used in HD_PULSE
        MuxGeom         = '';                       % Geometry of the muxing blocks (i.e. active elements)
        MuxSize                                     % Size of each mux'ing block [nElmtsAz nElmtsEl]
        ChannelIndex                                % Mapping of channel number for all the active elements
    end


    properties (Dependent, SetAccess = protected)
        elSizeEl                                    % Height of element [m]
        elSizeAz                                    % Width of element [m]
        xdcSizeAz                                   % Total transducer length
        xdcSizeEl                                   % Total transducer height
        nMuxAz                                      % Number of Mux~ing groups in az
        nMuxEl                                      % Number of Mux~ing groups in el
    end


    methods
        % =================================================================
        function probe = Probe(probe_name, MuxGeom, MuxSize, usePinout, debug_mode)
            % Probe
            % Probe(probe_name, MuxGeom, MuxSize, usePinout, debug_mode)
            %
            % Pedro Santos (KUL, 2017)

            % Parse inputs
            probe.name = probe_name;
            if exist('MuxGeom', 'var') && ~isempty(MuxGeom)
                probe.MuxGeom = MuxGeom;
            end
            if exist('MuxSize', 'var') && ~isempty(MuxSize)
                probe.MuxSize = MuxSize;
            end
            if exist('usePinout', 'var') && ~isempty(usePinout) && usePinout
                probe.usePinout = usePinout;
            end
            if exist('debug_mode', 'var') && ~isempty(debug_mode)
                probe.debug_mode = debug_mode;
            end

            % Define basic layout (gridded array)
            probe = createGridLayout(probe);

            % Calculate element positions
            probe = computeElementPositions(probe); % [nAz nEl [x|y|z]]

            % Remove elements from gridded layout and assign muxing size, if needed
            probe = modelProbeFromGriddedElements(probe);

            % Assign correct element position to HD-Pulse channel
            probe = applyConnectorPinout(probe);   % TODO: move to exportFocalLaws (since it's only required there) once probe is a class

            % Contruct muxing/channel table (if needed)
            probe = applyMuxingLayout(probe);

        end


        % =================================================================
        function is_equal = isEqualProbe(probe1, probe2)
            % This function can be used to compare if probe1 and probe2 have the same properties
            % This is used for debug purposes to make sure the new Probe class works as the old function
            % NOTE: first input must be a Probe object, the second one may be either a Probe obj or a struct.
            %
            % Pedro Santos (KUL, 2017)

            is_equal    = 1;
            diff_fields = {};

            % Check all fields in Probe1
            fields_pr1 = fieldnames(probe1);
            for field_i = fields_pr1'
                if isfield(probe2, field_i{1}) || isprop(probe2, field_i{1})
                    if isequal(probe1.(field_i{1}), probe2.(field_i{1}))
                        is_equal = is_equal && 1;
                    else
                        is_equal = is_equal && 0;
                        diff_fields(end+1,:) = {field_i{1}, probe1.(field_i{1}), probe2.(field_i{1})}; % Add differences to a list, to report at the end
                    end
                else
                    fprintf(' * Attribute %s not found in probe 1!\n', field_i{1})
                end
            end


            % Check all fields in Probe2
            fields_pr2 = fieldnames(probe2);
            for field_i = fields_pr2'
                if isfield(probe1, field_i{1}) || isprop(probe1, field_i{1})
                    if isequal(probe1.(field_i{1}), probe2.(field_i{1}))
                        is_equal = is_equal && 1;
                    else
                        is_equal = is_equal && 0;
                        if ~ismember(field_i{1}, diff_fields(:,1)) % If this field was still not listed, add it
                            diff_fields(end+1,:) = {field_i{1}, probe1.(field_i{1}), probe2.(field_i{1})};
                        end
                    end
                else
                    fprintf(' * Attribute %s not found in probe 2!\n', field_i{1})
                end
            end


            % Print the differences
            if ~is_equal
                fprintf('** The following fields are different between two probes compared:\n'), fprintf('\t%s\n', diff_fields{:,1})
                %                fprintf('\t\tField Name:\t\t\t\t Value Probe1: \t\t\t\t Value Probe1: \n')
                %                for ii = 1:size(diff_fields, 1)
                %                    fprintf('\t\t%s \t\t\t %s \t\t\t%s\n', diff_fields{ii, 1},  Utils.iif(ischar(diff_fields{ii, 2}), diff_fields{ii, 2}, num2str(diff_fields{ii, 2})),  Utils.iif(ischar(diff_fields{ii, 3}), diff_fields{ii, 3}, num2str(diff_fields{ii, 3})))
                %                end
            end
        end



        % =================================================================
        function probe = applyMuxingLayout(probe, new_mux_layout)

            if nargin == 2
                probe.MuxGeom = new_mux_layout;
            end


            % Create Channel mapping
            if ~isprop(probe, 'ChannelIndex') || isempty(probe.ChannelIndex)
                probe = createElmtMuxingIndex(probe); % assign muxing index (i.e. all elemenents inside muxer get the same index)
            end

            if any(probe.MuxSize > 1)
                % Potentially kill elements inside muxer
                if ~isempty(probe.MuxGeom)
                    probe = muxingAreaMod(probe);
                end

                % Compute Muxing centre coordinates
                probe = computeChannelCenters(probe);    % recompute element coordinates based on the muxing centroids
            end
        end



        % =================================================================
        function [probe] = muxingAreaMod(probe, MuxGeom_str)
            % note: This is meant for a muxing scheme of size 1xN (az x el)

            ApoutIndex = zeros(size(probe.ChannelIndex));
            IndexMax   = max(probe.ChannelIndex(:));
            % Count how many elements exist in each Muxer (if the probe was already modified, it may not be probe.MuxSize anymore)
            tmp_channel_nr = (probe.ChannelIndex(probe.ChannelIndex>0));
            muxSize    =  max(hist(tmp_channel_nr, numel(unique(tmp_channel_nr)))); % maximum number of elements per muxer


            % Configure how aperture changes from mux to mux
            if nargin == 2
                probe.MuxGeom = MuxGeom_str; % when muxingAreaMod is called outside Probe construction, MuxGeom is an input
            end
            str_mode = strsplit(probe.MuxGeom, '_');
            switch(lower(str_mode{1}))
                case 'fixed'
                    d_index  = 0;                 % the same for all muxers
                    keepList = eval(str_mode{2}); % which elements should stay alive
                case 'increment'
                    d_index  = eval(str_mode{2});       % value to increment
                    keepList = eval(str_mode{3}); % which elements should stay alive
                case 'rand'
                    if strcmpi(str_mode{2}, 'rand')
                        d_index  = randi(muxSize+1)-1;    % number of elements to kill is also random (allow no killing as well)
                    else
                        d_index  = eval(str_mode{2});  % number of elements to keep
                    end
                    muxElmKill = randperm(muxSize);         % generate random indexes from 1 to muxSize
                    keepList = muxElmKill(1:d_index);       % keep the first N elements active
                otherwise
                    return % Nothing to do here
            end


            muxElmKill  = setxor(1:muxSize, keepList);       % elements to kill (all but the ones contained in elmList)

            validIndex = 0;
            for i = 1:IndexMax % test all possible element indexes

                iMux     = (probe.ChannelIndex == i); % logical index matrix to current muxer
                iSapTmpV = probe.ChannelIndex(iMux); % Vector with current sap elements (index)...


                % Re-assign channel index to this muxer
                if numel(muxElmKill) < numel(iSapTmpV)
                    validIndex = validIndex + 1; % Increment channel number only if there's at least 1 element active
                end
                iSapTmpV(:) = validIndex;
                iSapTmpV(muxElmKill) = 0;


                % prepare next round of killings
                if strcmpi(str_mode{1}, 'rand')
                    % regenerate kill list randomly
                    muxElmKill = randperm(muxSize);     % generate random indexes from 1 to muxSize
                    if strcmpi(str_mode{2}, 'rand')
                        d_index = randi(muxSize+1)-1; % number of elements to kill is also random
                    end
                    muxElmKill = muxElmKill(d_index+1:end); % keep the last N elements, put the others on the kill list
                else
                    index_increment = d_index;
                    muxElmKill      = mod(muxElmKill + index_increment - 1, muxSize) + 1;  % increment RndV
                end

                probe.ChannelIndex(iMux) = iSapTmpV;

            end

            % Prune channel index (in case all elements wee deactivated in any mux)
            if max(probe.ChannelIndex(:)) > numel(unique(probe.ChannelIndex))
                for i = 1:IndexMax % test all possible element indexes
                    iMux                 = (probe.ChannelIndex == i); % logical index matrix to current muxer
                    iSapTmpV             = probe.ChannelIndex(iMux); % Vector with current sap elements (index)...
                end
            end




            if probe.debug_mode
                plotGenericAperture(probe), title('Deactivating elements inside Muxing')
            end
        end


        % =================================================================
        function probe = computeChannelCenters(probe, channel_idx)

            if exist('channel_idx', 'var') && ~isempty(channel_idx)
                probe.ChannelIndex = channel_idx; % overwrite channel index
            end

            muxCenters = nan(probe.nMuxAz * probe.nMuxEl, 3);

            for mux_i = 1:max(probe.ChannelIndex(:))

                this_mux = (probe.ChannelIndex == mux_i);   % select elements assigned to channel i

                elmentX  = probe.elemenPos(:,:,1);          % get coordinates of elements in this muxer
                elmentY  = probe.elemenPos(:,:,2);
                elmentZ  = probe.elemenPos(:,:,3);

                muxCenters(mux_i, :) = [mean2(elmentX(this_mux)) mean2(elmentY(this_mux)) mean2(elmentZ(this_mux))]; % Compute centre of mass
            end


            % Plot original Muxing
            if probe.debug_mode
                Probe.plotGenericAperture(probe), title('Muxed probe - channel centre')
            end

            % Store mux centre coordinates
            probe.MuxerPos = reshape(muxCenters, probe.nMuxAz, probe.nMuxEl, 3);


            if probe.debug_mode
                nElms = probe.nMuxAz * probe.nMuxEl;
                hold all, h_c = plot(1000*probe.MuxerPos(1:nElms), 1000*probe.MuxerPos(nElms+(1:nElms)), 'rs', 'markerFaceColor', 'r', 'markersize', 6); % plot (Az, El)
                legend(h_c, 'Mux centers', 'location', 'southoutside', 'orientation', 'vertical')
            end

        end



        % =================================================================
        function [probe] = createElmtMuxingIndex(probe)
            % Assign channel number to the individual elements. The Az/El blocks are defined based
            % on the geometric layout (i.e. independent of pinout channel mapping)

            nAz = probe.nElementsAz;
            nEl = probe.nElementsEl;
            MuxSizeAz = probe.MuxSize(1);
            MuxSizeEl = probe.MuxSize(2);
            nMuxAz  = probe.nMuxAz;         % number of saps in Az-direction
            nMuxEl  = probe.nMuxEl;         % number of saps in El-direction


            probe.ChannelIndex  = zeros(nAz, nEl);     % index for active rows/columns

            % SAP start value
            nStartAz = max(floor( (nAz-nMuxAz*MuxSizeAz)/2 ),1);     % Starting index for 1st Mux (in case aperture size is not multiple of mux size)
            nStartEl = max(floor( (nEl-nMuxEl*MuxSizeEl)/2 ),1);


            % Contruct Muxing index
            no = 1;
            for iy = nStartEl : MuxSizeEl : nEl-MuxSizeEl+1,
                for ix = nStartAz : MuxSizeAz : nAz-MuxSizeAz+1,
                    probe.ChannelIndex(ix:ix+MuxSizeAz-1, iy:iy+MuxSizeEl-1) = no;
                    no = no + 1;
                end
            end


            % If the pinout was computed, the elements position was shuffled. So, shuffle the channels as well
            if isfield(probe, 'connectorPinout') || (isprop(probe, 'connectorPinout') && ~isempty(probe.connectorPinout))
                probe.ChannelIndex(:) = probe.ChannelIndex(probe.connectorPinout);
            end


            % debug plot
            if probe.debug_mode
                Probe.plotGenericAperture(probe)
                title('Muxed probe')
            end

        end

        % =================================================================
        function probe = SetFlexibleArrayShape(probe,thetaRadians_LHS,thetaRadians_RHS)
            arcLength  = probe.nElementsAz*probe.pitchAz;
            rotOffset = 3*pi/2;
            midElement = int32((probe.nElementsAz)/2);
            x_dim = linspace(0,probe.pitchAz*(probe.nElementsAz-1),probe.nElementsAz);
            x_dim = x_dim - mean(x_dim);

            if thetaRadians_LHS == 0
                probe.elemenPos(1:midElement,1,1) = x_dim(1:midElement);
            else
                radius_LHS = arcLength/(thetaRadians_LHS*2);
                th_LHS = linspace(rotOffset - thetaRadians_LHS,rotOffset + thetaRadians_LHS,probe.nElementsAz);
                probe.elemenPos(1:midElement,1,1) = radius_LHS * cos(th_LHS(1:midElement));
                probe.elemenPos(1:midElement,1,3) = radius_LHS * (sin(th_LHS(1:midElement)) + 1);
            end
            if thetaRadians_RHS == 0
                probe.elemenPos(midElement+1:end,1,1) = x_dim(midElement+1:end);
            else
                radius_RHS = arcLength/(thetaRadians_RHS*2);
                th_RHS = linspace(rotOffset - thetaRadians_RHS,rotOffset + thetaRadians_RHS,probe.nElementsAz);
                probe.elemenPos(midElement+1:end,1,1) = radius_RHS * cos(th_RHS(midElement+1:end));
                probe.elemenPos(midElement+1:end,1,3) = radius_RHS * (sin(th_RHS(midElement+1:end)) + 1);
            end
        end

    end

    methods (Access = protected)

        % =================================================================
        function probe = createGridLayout(probe)
            switch(probe.name)
                case '1D Array'
                    probe.nElementsAz    = 64;
                    probe.nElementsEl    = 1;
                    probe.pitchAz        = 22e-5;
                    probe.pitchEl        = 12e-3;
                case '1D Vermon'
                    probe.nElementsAz    = 64;
                    probe.nElementsEl    = 1;
                    probe.pitchAz        = 30e-5;
                    probe.pitchEl        = 12e-3;
                case '1D Array 512'
                    probe.nElementsAz    = 512;
                    probe.nElementsEl    = 1;
                    probe.pitchAz        = 22e-5;
                    probe.pitchEl        = 12e-3;
                case '1D Array 1024'
                    probe.nElementsAz    = 1024;
                    probe.nElementsEl    = 1;
                    probe.pitchAz        = 10e-5;
                    probe.pitchEl        = 12e-3;
                case '1D Flex Olympus 128'
                    probe.nElementsAz    = 128;
                    probe.nElementsEl    = 1;
                    probe.pitchAz        = 0.41e-3;
                    probe.pitchEl        = 10e-3;
                case '1D Fraunhofer IBMT'
                    probe.nElementsAz    = 128;
                    probe.nElementsEl    = 1;
                    probe.pitchAz        = 0.3e-3;
                    probe.pitchEl        = 10e-3;
                case '2D Fraunhofer IBMT Matrix array'
                    probe.nElementsAz    = 11;
                    probe.nElementsEl    = 11;
                    probe.pitchAz        = 2.8125e-3;
                    probe.pitchEl        = 2.8125e-3;
                case 'Dual Linear Flex Olympus 128'
                    probe.nElementsAz    = 64;
                    probe.nElementsEl    = 2;
                    probe.pitchAz        = 0.3e-3;
                    probe.pitchEl        = 3e-3;
                    probe.kerfAz         = 0.08e-3;
                    probe.kerfEl         = 0.25e-3;
                case '2D Array'
                    probe.nElementsAz    = 33;
                    probe.nElementsEl    = 33;
                    probe.pitchAz       = 30e-5;
                    probe.pitchEl       = 30e-5;
                    probe.el_focus       = 15/1000;
                    probe.type           = 'xdc_2d_array';
                    probe.nSubEl         = probe.nSubAz;
                case '2D Vermon 1024'
                    probe.nElementsAz    = 32;
                    probe.nElementsEl    = 35;
                    probe.pitchAz        = 30e-5;
                    probe.pitchEl        = 30e-5;
                    probe.el_focus       = 80/1000;
                    probe.type           = 'xdc_2d_array';
                case '2D Vermon 256'
                    probe.nElementsAz    = 16;
                    probe.nElementsEl    = 17;
                    probe.pitchAz        = 30e-5;
                    probe.pitchEl        = 30e-5;
                    probe.el_focus       = 80/1000;
                    probe.type           = 'xdc_2d_array';
                case {'2D Vermon', '2D Vermon ControllerChassis', '2D Vermon FollowerChassis', '2D Vermon 32x8 - 1','2D Vermon 32x8 - 2', '2D Vermon 32x8 - 3', '2D Vermon 32x8 - 4', '2D Vermon 4x32 - 1', '2D Vermon 4x32 - 2', '2D Vermon 4x32 - 3', '2D Vermon 4x32 - 4', '2D Vermon 4x32 - 5', '2D Vermon 4x32 - 6', '2D Vermon 4x32 - 7', '2D Vermon 4x32 - 8', '2D Vermon 4x32 - noElev'}
                    % NOTE: All but '2D Vermon' are meant to be used in Single Chassis config
                    probe.nElementsAz    = 32;
                    probe.nElementsEl    = 35;
                    probe.pitchAz       = 30e-5;
                    probe.pitchEl       = 30e-5;
                    probe.el_focus       = 15/1000;
                    probe.type           = 'muxed_2D_array';

                case '1D 1024 Muxed'
                    probe.nElementsAz    = 1024;
                    probe.nElementsEl    = 1;
                    probe.pitchAz       = 30e-5;
                    probe.pitchEl       = 30e-5;
                    probe.kerfAz         = 5e-5; % Distance between transducer elements [m]
                    probe.kerfEl         = 5e-5; % Distance between transducer elements [m]
                    probe.el_focus       = 15/1000;
                    probe.type           = 'xdc_2d_array';
                    probe.MuxGeom        = MuxGeom;

                otherwise
                    % keyboard
                    if exist(probe.name, 'file')==2 % it is indicated a filename;
                        [ConfigFileData]=ReadConfigurationFile(probe.name);
                        probe.nElementsAz    = ConfigFileData.CrystalNr; % Number of Az elements
                        probe.nElementsEl    = 1;  % Number of El elements
                        probe.pitchAz        = ConfigFileData.Pitch;
                        probe.pitchEl        = ConfigFileData.Width;
                        probe.kerfAz         = ConfigFileData.Width-ConfigFileData.Pitch; % Distance between transducer elements [m]
                        probe.type           = 'xdc_linear_array';
                    else
                        disp('Probe name not recognized. Valid names are:')
                        cellfun(@(x) fprintf('\t* %s\n', x(2:end-1)), getValidSwitchCases());
                        return
                    end
            end

        end

        % =================================================================
        function probe = computeElementPositions(probe)
            % Computed the element coordinates from pitchAz/El and nElementsAz/El.
            % probe.elemenPos is outputed in the following format [nAz nEl [x|y|z]].

            d_az = probe.pitchAz*((1:probe.nElementsAz)-1-((probe.nElementsAz-1)/2));
            d_el = probe.pitchEl*((1:probe.nElementsEl)-1-((probe.nElementsEl-1)/2));
            d_dep = 0; % ROC not supported yet

            [azPos, elPos, depPos] = ndgrid(d_az, d_el, d_dep');

            probe.elemenPos = cat(3, azPos, elPos, depPos); % [nAz nEl [x|y|z]]

            if probe.debug_mode
                nElms = probe.nElementsAz * probe.nElementsEl;
                figure, plot(1000*probe.elemenPos(1:nElms), 1000*probe.elemenPos(nElms+(1:nElms)), 'rs', 'markersize', 8), % plot (Az, El)
                title(sprintf('Gridded Probe: %g x%g elements', probe.nElementsAz, probe.nElementsEl)), xlabel('Azimuth [mm]'), ylabel('Elevation [mm]'), axis equal tight
            end

        end

        % =================================================================
        function probe = modelProbeFromGriddedElements(probe)
            % modelProbeFromGriddedElements takes as input a rectangular gridded transducer and
            % kills aperture elements to model the desired transducer layout
            %
            % Pedro Santos (KUL, 2017)

            switch(probe.name)
                case '2D Vermon' % Kill rows (9, 18 & 27)
                    probe = killRowsAndColumns(probe, [9 18 27], []);
                case '2D Vermon ControllerChassis' % Kill top half of the probe
                    probe = killRowsAndColumns(probe, [9 18:35], []);
                case '2D Vermon FollowerChassis'
                    probe = killRowsAndColumns(probe, [1:18 27], []);  % Kill top half of the probe
                case '2D Vermon 32x8 - 1'
                    probe = killRowsAndColumns(probe, [9:35], []);
                case '2D Vermon 32x8 - 2'
                    probe = killRowsAndColumns(probe, [1:9 18:35], []);
                case '2D Vermon 32x8 - 3'
                    probe = killRowsAndColumns(probe, [1:18 27:35], []);
                case '2D Vermon 32x8 - 4'
                    probe = killRowsAndColumns(probe, [1:27], []);
                case {'2D Vermon 4x32 - 1', '2D Vermon 4x32 - noElev'}
                    probe = killRowsAndColumns(probe, setxor([1 5 10 14], [1:35]), []);
                case '2D Vermon 4x32 - 2'
                    probe = killRowsAndColumns(probe, setxor([2 6 11 15], [1:35]), []);
                case '2D Vermon 4x32 - 3'
                    probe = killRowsAndColumns(probe, setxor([3 7 12 16], [1:35]), []);
                case '2D Vermon 4x32 - 4'
                    probe = killRowsAndColumns(probe, setxor([4 8 13 17], [1:35]), []);
                case '2D Vermon 4x32 - 5'
                    probe = killRowsAndColumns(probe, setxor([19 23 28 32], [1:35]), []);
                case '2D Vermon 4x32 - 6'
                    probe = killRowsAndColumns(probe, setxor([20 24 29 33], [1:35]), []);
                case '2D Vermon 4x32 - 7'
                    probe = killRowsAndColumns(probe, setxor([21 25 30 34], [1:35]), []);
                case '2D Vermon 4x32 - 8'
                    probe = killRowsAndColumns(probe, setxor([22 26 31 35], [1:35]), []);
                case '2D Vermon 1024' % Kill rows (9, 18 & 27)
                    probe = killRowsAndColumns(probe, [9 18 27], []);
                case '2D Vermon 256'
                    probe = killRowsAndColumns(probe, 9, []);
            end

            % Set the muxing size, if not given by the user
            if isempty(probe.MuxSize)
                if strcmp(probe.name,'2D Vermon')
                    probe.MuxSize = [1 4];
                else
                    probe.MuxSize = [1 1];
                end
            end
        end

        % =================================================================
        function [probe] = killRowsAndColumns(probe, RowKillList, ColumnKillList)

            if probe.debug_mode
                nElms = probe.nElementsAz * probe.nElementsEl;
                figure, plot(1000*probe.elemenPos(1:nElms), 1000*probe.elemenPos(nElms+(1:nElms)), 'rs', 'markersize', 8), % plot (Az, El)
            end

            % Kill rows/columns as requested
            probe.elemenPos(:, RowKillList, :) = [];
            probe.elemenPos(:, :, ColumnKillList) = [];

            % Recount number of elements
            probe.nElementsAz = size(probe.elemenPos, 1);
            probe.nElementsEl = size(probe.elemenPos, 2);

            if probe.debug_mode
                nElms = probe.nElementsAz * probe.nElementsEl;
                hold all, plot(1000*probe.elemenPos(1:nElms), 1000*probe.elemenPos(nElms+(1:nElms)), 'bx'), % plot (Az, El)
                title('Kill elements'), legend('Original Elements', 'Elements Kept'), xlabel('Azimuth [mm]'), ylabel('Elevation [mm]'), axis equal tight
            end
        end



        % =================================================================
        function probe = applyConnectorPinout(probe)


            if ~probe.usePinout
                % If no pinout is applied, use linear channel indexing
                probe.connectorPinout = [1:probe.nElementsAz*probe.nElementsEl];
                return
            end


            switch(probe.name)
                case '1D Vermon'

                    connector_idx(1:2:64) = 32:-1:1;
                    connector_idx(2:2:64) = 64:-1:33;

                case {'2D Vermon ControllerChassis', '2D Vermon FollowerChassis'}

                    connector_idx = [1:32    129:160   257:288   385:416  , ... % connector #1
                        33:64   161:192   289:320   417:448  , ... % connector #2
                        65:96   193:224   321:352   449:480  , ... % connector #3
                        97:128  225:256   353:384   481:512];      % connector #4

                case '2D Vermon'

                    %                     connector_idx = [1:32    129:160   257:288   385:416  , ... % connector #1
                    %                         33:64   161:192   289:320   417:448  , ... % connector #2
                    %                         65:96   193:224   321:352   449:480  , ... % connector #3
                    %                         97:128  225:256   353:384   481:512];      % connector #4
                    %
                    %                     connector_idx = [connector_idx 512+connector_idx]; % add 2nd chassis

                    connector_idx(1:256) = bsxfun(@plus, 1:32, 32*[0:4:31]')';      % connector 1
                    connector_idx(257:512) = bsxfun(@plus, 1:32, 32*[1:4:31]')';    % connector 2
                    connector_idx(513:768) = bsxfun(@plus, 1:32, 32*[2:4:31]')';    % connector 3
                    connector_idx(769:1024) = bsxfun(@plus, 1:32, 32*[3:4:31]')';   % connector 4


                otherwise
                    connector_idx = [1:probe.nElementsAz*probe.nElementsEl];
            end

            % Now reorder probe elements
            probe.connectorPinout = connector_idx;
            tmp_elem_pos          = reshape(probe.elemenPos, [], 3);  % [Naz*Nel x 3]
            tmp_elem_pos          = tmp_elem_pos(probe.connectorPinout, :);
            probe.elemenPos       = reshape(tmp_elem_pos, probe.nElementsAz, probe.nElementsEl, 3);

        end

        % =================================================================
        function probe = recentreProbe(probe)
            probe.elemenPos(:, 2, : ) = probe.elemenPos(:, 1, : );
            probe.elemenPos(:, 3, : ) = probe.elemenPos(:, 1, : );
            probe.elemenPos(:, 4, : ) = probe.elemenPos(:, 1, : );
        end


        % =================================================================
        function probe = clearMuxingLayout(probe)
            % clearMuxingLayout reverts the probe back to fully wired array, removing all muxing settings
            %
            % Pedro Santos (KUL, 2017)

            keyboard % not tested

            % Define basic layout (gridded array)
            probe = createGridLayout(probe);

            % Calculate element positions
            probe = computeElementPositions(probe); % [nAz nEl [x|y|z]]

            % Remove elements from gridded layout, if needed
            probe = modelProbeFromGriddedElements(probe);
        end

    end

    methods

        % =================================================================
        % Set/Get methods
        function elem_az = get.elSizeAz(probe)
            elem_az = probe.pitchAz - probe.kerfAz;
        end

        function elem_az = get.elSizeEl(probe)
            elem_az = probe.pitchEl - probe.kerfEl;
        end

        function size_az = get.xdcSizeAz(probe)
            size_az = max(max(probe.elemenPos(:,:,1))) - min(min(probe.elemenPos(:,:,1))) + probe.elSizeAz;
        end

        function size_el = get.xdcSizeEl(probe)
            size_el = max(max(probe.elemenPos(:,:,2))) - min(min(probe.elemenPos(:,:,2))) + probe.elSizeAz;
        end

        function n_mux_az = get.nMuxAz(probe)
            n_mux_az = ceil(probe.nElementsAz / probe.MuxSize(1));
        end

        function n_mux_el = get.nMuxEl(probe)
            n_mux_el = ceil(probe.nElementsEl / probe.MuxSize(2));
        end

    end

    methods (Static)
        % =================================================================
        function plotGenericAperture(probe, plotMode, delays, h_fig, varargin)

            if ~exist('h_fig', 'var') || isempty(h_fig)
                h_fig = figure;
                h_axes = gca;
            else
                if isa(h_fig, 'matlab.ui.Figure')  % we passed a figure
                    h_axes = gca;
                elseif isa(h_fig, 'matlab.graphics.axis.Axes') % we passed an axes
                    h_axes = h_fig;
                    h_fig = get(h_axes, 'parent');
                end

            end
            if ~exist('plotMode', 'var') || isempty(plotMode)
                plotMode = 1; % nomal mode, color coding
            end
            if plotMode == 2 && (~exist('delays', 'var') || isempty(delays))
                plotMode = 1; % cannot plot delays
                disp('"plotGenericAperture: Cannot plot delays, because no delays were input. Plotting channel index instead.')
            end


            % element sizes
            if isnumeric(probe)
                % We have Field II aperture
                keyboard
            elseif isstruct(probe) || isa(probe,'Probe')
                % We have Matlab struct
                X = [-probe.elSizeAz -probe.elSizeAz probe.elSizeAz  probe.elSizeAz]/2; % corner order: SW, NW, NE, SE
                Y = [-probe.elSizeEl  probe.elSizeEl probe.elSizeEl -probe.elSizeEl]/2;
            else
                keyboard
            end

            % construce element borders
            Xel = bsxfun(@plus, reshape(probe.elemenPos(:,:,1), 1, []),  X');
            Yel = bsxfun(@plus, reshape(probe.elemenPos(:,:,2), 1, []),  Y');
            MyAxis = 1000*[min(Xel(:)) max(Xel(:)) min(Yel(:)) max(Yel(:))];



            % Color code channel number
            switch(plotMode)
                case 1
                    % channel number
                    if isfield(probe, 'ChannelIndex') || (isprop(probe, 'ChannelIndex') && ~isempty(probe.ChannelIndex))
                        channel_color = reshape(probe.ChannelIndex, 1, []); % If we have muxing
                    else
                        channel_color = 1:size(Xel, 2); % In normal arrays
                    end
                case 2
                    % delays
                    channel_color = NaN*reshape(probe.ChannelIndex, 1, []); % If we have muxing
                    channel_color(probe.ChannelIndex>0) = reshape(delays, 1, []);
                case 3
                    % radius
                    mux_pos = [NaN sqrt(reshape(probe.MuxerPos(:,:,1), 1, []).^2 + reshape(probe.MuxerPos(:,:,2), 1, []).^2)];
                    channel_color = mux_pos(reshape(probe.ChannelIndex, 1, [])+1); % If we have muxing
                otherwise
                    keyboard
            end



            C = [1; 1; 1; 1] * channel_color; % hardware channel number

            % Colormap
            if plotMode == 2
                my_cmap = jet(256);
            else
                if min(channel_color(:)) == 0
                    my_cmap = [1 1 1; lines(max(channel_color(:))-1)]; % use white color for deactivated elements
                else
                    my_cmap = [jet(max(channel_color(:)))];
                end
            end

            figure(h_fig), axes(h_axes)
            fill(1000*Xel, 1000*Yel, C)
            xlabel('Azimuth [mm]'), ylabel('Elevation [mm]'), axis equal, axis(MyAxis), colormap(my_cmap);


        end
    end

end
