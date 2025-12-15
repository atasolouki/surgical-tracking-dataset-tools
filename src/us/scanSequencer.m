classdef scanSequencer < dynamicprops
    %UNTITLED5 Summary of this class goes here
    %   Note: This has been prorted from the non-class function and could
    %   use some optimization, including removal of some alien fields
    %   like 'nCompoundAz' and 'nFrames'

    properties
        txFocalDepth
        f0
        nTxAz
        nTxEl
        nMLAs
        nMLTs                = [1 1];
        PRF
        txEvents
        compoundingMode
        DWSlidingTx
        nPackets
        openingAngle
        SectorTilt
        Doppler
        Muxing
        chSampleTime
        t0_tx
        nFrames
        fDemod
    end

    properties (Access = protected)
        is_initialized          = 0;    % Used by the set methods
        tx_angles_az                    % Do we really need this?
        tx_angles_el                    % Do we really need this?
        MLA_spread_az
        MLA_spread_el
        Probe
    end

    properties (Dependent = true)
        ImageSector
        nScanLines
    end

    methods


        function obj = scanSequencer(varargin)
            % Create basic geometry
            if nargin < 3
                if numel(varargin{1}.file_list.Datasets) > 0
                    obj = scanSequencer.constructFromFile(obj, varargin{:});
                    return
                else
                    obj = scanSequencer.constructFromHDUnpacker(obj, varargin{:});
                end
            else
                obj = scanSequencer.constructFromInputs(obj, varargin{:});
            end
            obj = validateSequence(obj);


            % Setup basic Tx/Rx layout
            if nargin < 3
                obj = setupTxAngles(obj, varargin{1}.cfg_meta);
            else
                obj = setupTxAngles(obj);
            end
            obj = setupRxAngles(obj);

            % MLT
            if any(obj.nMLTs > 1)
                obj = makeSequenceMLT('regular', obj, obj.nMLTs);
            end

            % Adapt sequence
            obj = setupCompounding(obj);
            obj.is_initialized = 1;

            % If needed, setup Doppler packets
            if ~isempty(obj.Doppler)
                obj = setupDoppler(obj);
            end

        end % END scanSequencer

        function setCartesianRXLinePositions(obj,xLocs,yLocs)
            [X,Y] = meshgrid(xLocs,yLocs);
            obj.txEvents(1).line_pos = cat(3,X',Y');
        end

        function obj = setupMLTSequence(obj,nMLTs)
            if ~all(nMLTs == 1)
                obj.nMLTs = nMLTs;
                txEventslocal = obj.txEvents;
    
                if obj.compoundingMode == 0
                    nTxAzlocal = obj.nTxAz;
                    nTxEllocal = obj.nTxEl;
                    obj.nTxAz = ceil(nTxAzlocal / nMLTs(1));
                    obj.nTxEl = ceil(nTxEllocal / nMLTs(2));
                else
                    nTxAzlocal = length(obj.tx_angles_az);
                    nTxEllocal = length(obj.tx_angles_el);
                    obj.nPackets = numel(txEventslocal)/nMLTs(1)/nMLTs(2);
                end

                for n_tx = 1 : numel(txEventslocal)/nMLTs(1)/nMLTs(2)
                    az_idx = [0:nTxAzlocal/obj.nMLTs(1):nTxAzlocal-1]+1;
                    el_idx = [0:nTxEllocal/obj.nMLTs(2):nTxEllocal-1]+1;
                    idx_grid = reshape(1:numel(txEventslocal),nTxAzlocal,nTxEllocal)';
                    beams_in_this_tx = idx_grid(el_idx,az_idx) + n_tx - 1;
    
                    obj.txEvents(n_tx).txAngle = cat(2, txEventslocal(beams_in_this_tx).txAngle);     % MLTs will go column-wise
                    obj.txEvents(n_tx).txFocus = cat(1, txEventslocal(beams_in_this_tx).txFocus);     % MLTs will go row-wise here (TODO: adapt scanSequencer to have it also columns-wise)
                    obj.txEvents(n_tx).rxAngles = cat(2, txEventslocal(beams_in_this_tx).rxAngles);
                end
            obj.txEvents(n_tx+1:end) = []; % remove additional events
            end
        end

        function [] = plot(obj, fig_h)
            % SCANSEQUENCER.PLOT plots the transmit and receive focal positions of a sequence

            if ~exist('fig_h', 'var') || isempty(fig_h), fig_h = figure; end

            rx_max_depth = .06; % maximum depth for Rx lines % TODO: take if from maximum scanning depth
            n_rx_beams   = prod(obj.nMLAs);
            if (isfield(obj, 'nMLTs') || isprop(obj, 'nMLTs')) && prod(obj.nMLTs) > 1
                n_tx_beams = prod(obj.nMLTs);
                n_rx_beams = prod(obj.nMLTs);
            else
                n_tx_beams = 1;
            end

            figure(fig_h), hold all
            axis auto, axis equal tight, set(gca, 'zdir', 'reverse')
            co = get(0, 'defaultAxesColorOrder'); % take color order to plot Tx/Rx in the same color for a given txEvent
            nco = size(co, 1);
            xlabel('Azimuth [mm]'), ylabel('Elevation [mm]'), zlabel('Range [mm]')
            for tx_i = 1:numel(obj.txEvents)

                % Transmit focus
                tx_focus = obj.txEvents(tx_i).txFocus * 1000';
                plot3([zeros(1, n_tx_beams); tx_focus(:,1)'], [zeros(1, n_tx_beams); tx_focus(:,2)'], [zeros(1, n_tx_beams); tx_focus(:,3)'], '-x', 'color', co(mod(tx_i-1, nco)+1, :))
                %                 scatter(tx_focus(:,1)', tx_focus(:,2)', 40, repmat(co(tx_i, :), size(tx_focus, 1), 1), 'filled') % top view

                % Receive focus
                rx_angles_event = obj.txEvents(tx_i).rxAngles';
                rx_focus        = Utils.beam2cart(rx_angles_event(:,1), rx_angles_event(:,2), rx_max_depth*1000);
                plot3([zeros(1, n_rx_beams); rx_focus(:,1)'], [zeros(1, n_rx_beams); rx_focus(:,2)'], [zeros(1, n_rx_beams); rx_focus(:,3)'], '--o', 'color', co(mod(tx_i-1, nco)+1, :))

                if any(tx_focus(:,2))
                    view([0 90]), axis equal tight
                else
                    view([0 0]), axis equal tight
                end
                title(sprintf('Tx %g/%g', tx_i, numel(obj.txEvents)))
                pause(0.1)
            end
            hold off
            legend('Tx beam', 'Rx beam', 'location', 'NorthEast')



        end % END plot
    end

    methods (Access = protected)
        function obj = recreateTxRxSequence(obj)
            obj = setupTxAngles(obj);
            obj = setupRxAngles(obj);
            obj = setupCompounding(obj);
        end

        function obj = recreateRxSequence(obj)
            obj = setupRxAngles(obj);
            if ~isempty(obj.Doppler)
                obj = setupDoppler(obj);
            end
        end


        %% Main sequencer
        function obj = setupTxAngles(obj, HDP_meta)
            % Tx angles
            if exist('HDP_meta', 'var') && ~isempty(HDP_meta)
                obj.tx_angles_az  = HDP_meta.Firing_Az;
                obj.tx_angles_el  = HDP_meta.Firing_El;
            else
                delta_Tx          = obj.ImageSector ./ max([obj.nTxAz obj.nTxEl], 1);
                obj.tx_angles_az  = linspace(-obj.ImageSector(1)/2+delta_Tx(1)/2, obj.ImageSector(1)/2-delta_Tx(1)/2, obj.nTxAz) + obj.SectorTilt(1);
                obj.tx_angles_el  = linspace(-obj.ImageSector(2)/2+delta_Tx(2)/2, obj.ImageSector(2)/2-delta_Tx(2)/2, obj.nTxEl) + obj.SectorTilt(2);
            end

            % Construct Tx events
            for tx_el = 1 : obj.nTxEl
                for tx_az = 1 : obj.nTxAz

                    tx_i = (tx_el-1)*obj.nTxAz + tx_az; % current Tx event number

                    % Tx Angle
                    obj.txEvents(tx_i).txAngle = [obj.tx_angles_az(tx_az); obj.tx_angles_el(tx_el)];

                    % Tx Focus
                    if ~obj.DWSlidingTx
                        obj.txEvents(tx_i).txFocus = Utils.beam2cart(obj.tx_angles_az(tx_az), obj.tx_angles_el(tx_el), obj.txFocalDepth);
                        obj.txEvents(tx_i).txAngle = [obj.tx_angles_az(tx_az); obj.tx_angles_el(tx_el)];
                    else
                        if obj.Probe.nElementsEl > 1, keyboard, end % not implemented
                        if exist('HDP_meta', 'var') && any(strcmp(HDP_meta.ScanMode, {'Hadamard DW', 'Sliding DW'}))
                            tx_translation = obj.Probe.pitchAz*(HDP_meta.CentralElement(tx_i)-HDP_meta.CrystalNr/2);
                        else
                            if obj.nTxAz==1
                                tx_translation = 0;
                            else
                                centralElement = floor(obj.DWSlidingTx/2)+1 + (tx_az-1)*floor((obj.Probe.nElementsAz-obj.DWSlidingTx)/(obj.nTxAz-1));
                                tx_translation = obj.Probe.pitchAz*(-(obj.Probe.nElementsAz-1)/2 + (centralElement-1));
                            end
                        end

                        obj.txEvents(tx_i).txFocus = [tx_translation 0 obj.txFocalDepth];
                        obj.txEvents(tx_i).txAngle = [0; obj.tx_angles_el(tx_el)]; % force azimuth angle to zero
                    end
                end %END tx_az
            end % END tx_el
            obj.txEvents(tx_i+1:end) = []; % make sure we delete remaining events
        end % END setupTxAngles

        function obj = setupRxAngles(obj)
            if all(obj.tx_angles_az) == 0
                tx_angles = cat(2, obj.txEvents.txAngle);
                obj.tx_angles_az = unique(tx_angles(1,:));
                obj.tx_angles_el = unique(tx_angles(2,:));
            end

            switch(obj.compoundingMode)
                case 0 % no compunding
                    delta_Tx(1)        = Utils.iif(numel(obj.tx_angles_az) > 1, mean(diff(obj.tx_angles_az)), obj.ImageSector(1));
                    delta_Tx(2)        = Utils.iif(numel(obj.tx_angles_el) > 1, mean(diff(obj.tx_angles_el)), obj.ImageSector(2));
                    delta_Rx           = delta_Tx ./ obj.nMLAs;
                    obj.MLA_spread_az  = linspace(-(delta_Tx(1)/2 - delta_Rx(1)/2), delta_Tx(1)/2 - delta_Rx(1)/2, obj.nMLAs(1));
                    obj.MLA_spread_el  = linspace(-(delta_Tx(2)/2 - delta_Rx(2)/2), delta_Tx(2)/2 - delta_Rx(2)/2, obj.nMLAs(2));
                case {1,2,3}
                    delta_Rx           = obj.ImageSector ./ obj.nMLAs;
                    obj.MLA_spread_az  = linspace(-(obj.ImageSector(1)/2 - delta_Rx(1)/2), obj.ImageSector(1)/2 - delta_Rx(1)/2, obj.nMLAs(1)) + obj.SectorTilt(1);
                    obj.MLA_spread_el  = linspace(-(obj.ImageSector(2)/2 - delta_Rx(2)/2), obj.ImageSector(2)/2 - delta_Rx(2)/2, obj.nMLAs(2)) + obj.SectorTilt(2);
                otherwise
                    warndlg(sprintf('Compounding mode %g not recognized in scanSequencer', obj.compoundingMode))
                    error('@scanSequencer: Compounding mode %g not recognized', obj.compoundingMode)
            end


            % Setup receive events
            for tx_el = 1 : obj.nTxEl
                for tx_az = 1 : obj.nTxAz
                    tx_i = (tx_el-1)*obj.nTxAz + tx_az;              % current Tx event number
                    for mla_el = 1:obj.nMLAs(2)
                        for mla_az = 1:obj.nMLAs(1)

                            rx_i = (mla_el-1)*obj.nMLAs(1) + mla_az; % current Rx line number

                            switch(obj.compoundingMode)
                                case{0}
                                    obj.txEvents(tx_i).rxAngles(:, rx_i) = obj.txEvents(tx_i).txAngle + [obj.MLA_spread_az(mla_az); obj.MLA_spread_el(mla_el)];
                                    if obj.DWSlidingTx ~= 0 % Sliding Focused Beam ie linear-stepped sequence
                                        locs = vertcat(obj.txEvents(:).txFocus)';
                                        xlocs = unique(locs(1,:),'rows');
                                        ylocs = unique(locs(2,:)','rows');
                                        [X,Y] = meshgrid(xlocs,ylocs);
                                        obj.txEvents(tx_i).line_pos = cat(3,X',Y');
                                    end
                                case{1,2,3}
                                    obj.txEvents(tx_i).rxAngles(:, rx_i) = [obj.MLA_spread_az(mla_az) obj.MLA_spread_el(mla_el)]';  %  we will reconstruct a full image for all firings
                                    if obj.txFocalDepth == 0
                                        % For plane wave compounding we set here the xy positions of the lines.
                                        if obj.nMLAs(1) > 1
                                            xLocs = linspace(obj.Probe.elemenPos(1,1,1),obj.Probe.elemenPos(end,1,1),obj.nMLAs(1));
                                        else
                                            xLocs = (obj.Probe.elemenPos(1,1,1)+obj.Probe.elemenPos(end,1,1))/2;
                                        end
                                        if obj.nMLAs(2) > 1
                                            yLocs = linspace(obj.Probe.elemenPos(1,1,2),obj.Probe.elemenPos(1,end,2),obj.nMLAs(2));
                                        else
                                            yLocs = (obj.Probe.elemenPos(1,1,2)+obj.Probe.elemenPos(1,end,2))/2;
                                        end
                                        [X,Y] = meshgrid(xLocs,yLocs);
                                        obj.txEvents(tx_i).line_pos = cat(3,X',Y');
                                    end

                            end
                        end % END mla_az
                    end % END mla_el
                    obj.txEvents(tx_i).rxAngles(:, rx_i+1:end) = []; % make sure we delete remaining events
                end % END tx_az
            end % END tx_el
        end % END setupRxAngles



        %%
        function obj = setupCompounding(obj)
            if obj.compoundingMode == 3      % Special case for Hadamard (ToDo: include Hadamard as MLTs)
                obj.txEvents = repmat(obj.txEvents, 1, obj.nTxAz);
                obj.nTxAz = obj.nTxAz^2;
                obj.nTxEl = obj.nTxEl^2;
                obj.nPackets = obj.nTxAz * obj.nTxEl;
            elseif obj.compoundingMode > 0   % For full-sector compunding, each packet has actually only 1 Tx
                obj.nPackets = obj.nTxAz * obj.nTxEl;
                obj.nTxAz = 1;
                obj.nTxEl = 1;
            elseif ~isempty(obj.Doppler)
                % do nothing. packets were already there
            else
                obj.nPackets = 1;
            end
        end


        %%
        function obj = setupDoppler(obj)
            debug_plot = 0;
            obj = DopplerProcessor.updateSequenceForDoppler(obj, obj.Doppler, debug_plot);
        end

        %%
        function obj = validateSequence(obj)
            if numel(obj.ImageSector) == 1
                obj.ImageSector = [obj.ImageSector 0]; % elevation opening angle null (2D scan)
            end
            if obj.nTxEl > 1 && (obj.nScanLines(2) == 1 || numel(obj.ImageSector) == 1 || obj.ImageSector(2) == 0)
                fprintf('\tFor elevation scanning, the following is required:\n\t\t* more than 1 Rx elevation line\n\t\t* positive opening angle in elevation.\n')
            end
            %we comment it out because compoundingMode should be 1 for a
            %single PW and DW to work (0° angle)
            % if obj.compoundingMode > 0 && obj.nTxAz*obj.nTxEl == 1
            %     disp('*** WARNING: Coherent compounding not performed, because nTxAz = nTxEl = 1')
            %     obj.compoundingMode = 0;
            % end
        end
    end

    %% Constructors
    methods (Static, Access = protected)
        function obj = constructFromFile(obj, varargin)
            % If a .mat dataset has been stored when exporting the .law, use it
            % The user was asked whether to use this file in recreateProbe(), so there's always the option to ignore it
            keyboard % needs to be adapted for class!!
            ChDataSet    = importdata(varargin{1}.file_list.Datasets{1});
            scanSequence = ChDataSet.meta.Sequence;
            scanSequence.txEvents(1:varargin{1}.cfg_meta.NArfPushes) = [];
        end


        function obj = constructFromHDUnpacker(obj, varargin)
            HDP_meta            = varargin{1}.cfg_meta;
            obj.txFocalDepth    = HDP_meta.focusDepth;
            obj.nTxAz           = size(HDP_meta.Firing_Az, 2);
            obj.nTxEl           = 1; % For HD-PULSE, do not distinguish between Az & El
            nImgLines           = [HDP_meta.NrOfImageLines 1];
            obj.nMLTs           = [HDP_meta.MLTnr 1];
            obj.openingAngle    = HDP_meta.SectorOpening;
            obj.DWSlidingTx     = Utils.iif(any(strcmp(HDP_meta.ScanMode, {'Sliding DW', 'Hadamard DW'})), 1, 0);
            obj.Probe           = varargin{1}.probe;
            obj.SectorTilt      = HDP_meta.SectorTilt;
            if isfield(HDP_meta, 'PRF'), obj.PRF = HDP_meta.PRF; end
            obj.Doppler = [];
            switch(HDP_meta.ScanMode)
                case {'MLT', 'Anatomical'}
                    obj.compoundingMode = 0;
                case {'Sliding DW', 'Tilting DW'}
                    obj.compoundingMode = 1;
                case 'Hadamard DW'
                    obj.compoundingMode = 1;
                otherwise
                    errordlg(sprintf('HD-PULSE scanning mode "%s" not recognized', HDP_meta.ScanMode), 'Generate Scan Sequence', 'replace');
                    error('HD-PULSE scanning mode "%s" not recognized', HDP_meta.ScanMode)
            end
            % Setup nMLAs
            if nImgLines == 0
                obj.nMLAs = [1 1];
            elseif obj.compoundingMode == 0
                obj.nMLAs = ceil(nImgLines ./ [obj.nTxAz obj.nTxEl]);
            else
                obj.nMLAs = nImgLines;     % reconstruct full image for all firings
            end
        end


        function obj = constructFromInputs(obj, varargin)
            varargin(nargin+1:10) = {[]}; % fill remaining inputs as empty
            obj.txFocalDepth    = varargin{1};
            obj.nTxAz           = varargin{2};
            obj.nTxEl           = varargin{3};
            obj.openingAngle    = varargin{4};
            nImgLines           = Utils.iif(nargin>4 & ~isempty(varargin{5}), varargin{5}, 0);
            obj.compoundingMode = Utils.iif(nargin>5 & ~isempty(varargin{6}), varargin{6}, 0);
            obj.DWSlidingTx     = Utils.iif(nargin>6 & ~isempty(varargin{7}), varargin{7}, 0);
            obj.Probe           = Utils.iif(nargin>7 & ~isempty(varargin{8}), varargin{8}, 0);
            obj.SectorTilt      = Utils.iif(nargin>8 & ~isempty(varargin{9}), varargin{9}, [0 0]);
            obj.Doppler         = Utils.iif(nargin>9 & ~isempty(varargin{10}), varargin{10}, []);
            % Setup nMLAs
            if numel(nImgLines) == 1, nImgLines = [nImgLines 1]; end
            if obj.compoundingMode == 0
                obj.nMLAs = ceil(nImgLines ./ [obj.nTxAz obj.nTxEl]);
            else
                obj.nMLAs = nImgLines;     % reconstruct full image for all firings
            end
        end


    end


    %% set/get methdos
    methods

        function obj = set.nTxAz(obj, val_in)
            assert(isreal(val_in) && isscalar(val_in), 'Value must be a real scalar.')
            obj.nTxAz = val_in;
            if obj.is_initialized
                obj = recreateTxRxSequence(obj);
            end
        end
        function obj = set.nTxEl(obj, val_in)
            assert(isreal(val_in) && isscalar(val_in), 'Value must be a real scalar.')
            obj.nTxEl = val_in;
            if obj.is_initialized
                obj = recreateTxRxSequence(obj);
            end
        end
        function obj = set.nMLAs(obj, val_in)
            assert(isreal(val_in) && (numel(val_in)==1 || numel(val_in)==2) , 'Value must be a real scalar (Az) or 2-element vector (Az,El).')
            if numel(val_in) == 1, val_in = [val_in 1]; end
            obj.nMLAs = val_in;
            if obj.is_initialized
                obj = recreateRxSequence(obj);
            end
        end
        function obj = set.openingAngle(obj, val_in)
            assert(isreal(val_in) && (numel(val_in)==1 || numel(val_in)==2) , 'Value must be a real scalar (Az) or 2-element vector (Az,El).')
            if numel(val_in) == 1, val_in = [val_in 0]; end
            obj.openingAngle = val_in;
        end
        function obj = set.SectorTilt(obj, val_in)
            assert(isreal(val_in) && (numel(val_in)==1 || numel(val_in)==2) , 'Value must be a real scalar (Az) or 2-element vector (Az,El).')
            if numel(val_in) == 1, val_in = [val_in 0]; end
            obj.SectorTilt = val_in;
        end
        function im_sector = get.ImageSector(obj)
            im_sector = obj.openingAngle + obj.SectorTilt;
        end
        function n_lines = get.nScanLines(obj)
            if obj.compoundingMode == 0
                n_lines = [obj.nTxAz obj.nTxEl] .* obj.nMLAs .* obj.nMLTs;
            else
                n_lines = obj.nMLAs;     % reconstruct full image for all firings
            end
        end

    end


end

