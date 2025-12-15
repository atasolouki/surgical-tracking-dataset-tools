classdef ULAOP_UnpackProcessor

    properties

        data_path               % Channel data path
        file_list               % List of all revevant file paths
        meta                    % Metadata needed to align with HDPulse code
        DataObj
        slice_idx
        data_type
        nGate
        probe                   % Probe object reconstructed from cfg metadata
        dateStamp               % Time exported in cfg file
        timeStamp               % Date exported in cfg file
        scanSequence            % Instance of the scanSequencer class
        Scanner = 'ULA-OP';     % Scanner name
    end

    methods (Static)
        function [ChDataset] = PrepareDataset(data_path,varargin)
            % PrepareDataset creates an (empty) ULA-OP channel dataset.
            %
            %    CH_SET = PrepareDataset(CHDATA_PATH, STREAM_VER), where CHDATA_PATH is the channel data path
            %    and STREAM_VER is an optional input which refers to the version of the streaming, if the
            %    user wants to override it. CH_SET is a structure with the
            %    proper metadafa fields (sequence, probe, etc), but with
            %    the data field empty.
            %
            % Marcus Ingram (KU Leuven, 2023)
            %
            % See also ULAOP_UnpackProcessor

            % Extract basic stream info
            ulaop_obj = ULAOP_UnpackProcessor(data_path,varargin{:});

            % Create ChData structure
            ChDataset = packDataset([], ulaop_obj.probe, ulaop_obj.scanSequence, [], ulaop_obj.meta.ch_data_time, ulaop_obj.meta.SamplingFreq, ulaop_obj.meta.c0, ulaop_obj.meta.f0); % empty data
            [ChDataset.meta.AcqInfo.Date,ChDataset.meta.AcqInfo.Time] = get_ulaop_acq_date_time(ulaop_obj);               % Get date/time stamps
            ChDataset.meta.AcqInfo.ScanMode = 'Anatomical';

            % Load ECG
            %             if isfield(hdp_obj.file_list, 'Ecg') && ~isempty(hdp_obj.file_list.Ecg)
            %                 [ChDataset.meta.Ecg.data, ChDataset.meta.Ecg.time, ChDataset.meta.Ecg.its] = ECGprocessor.loadSyncedECG(hdp_obj, hdp_obj.valid_FRD_frames);
            %                 ChDataset.meta.Ecg.time = ChDataset.meta.Ecg.time - ChDataset.meta.Ecg.time(1);
            %             end

            % Prepare reconstruction
            [ChDataset.meta.Beamforming, ChDataset.meta.PostProcessing] = setUpReconstructionStructure(ulaop_obj.meta);

            % Pass unpacker info as well
            ChDataset.meta.Unpacking     = ulaop_obj.meta;
            ChDataset.meta.file_list     = ulaop_obj.file_list;
            ChDataset.meta.ULAOP_DataObj = ulaop_obj.DataObj;
            ChDataset.meta.data_type     = ulaop_obj.data_type;
        end
    end


    methods

        function obj = ULAOP_UnpackProcessor(data_path,varargin) % constructor
            % ULAOP_UnpackProcessor create the ULAOP unpack structure with information about valid
            % metadata read from the uop file.
            %
            %    ULAOP_UnpackProcessor(CHDATA_PATH), where CHDATA_PATH is the channel data path
            %
            % Marcus Ingram (KU Leuven, 2023)
            obj.data_path = data_path;
            obj.slice_idx = 0;
            obj.data_type = 'RFPre';

            for ii = 1:2:length(varargin)
                if strcmp('slice_idx', varargin{ii})
                    obj.slice_idx = varargin{ii+1};
                elseif strcmp('data_type', varargin{ii})
                    obj.data_type = varargin{ii+1};
                elseif strcmp('scanSequence', varargin{ii})
                    obj.scanSequence = varargin{ii+1};
                elseif strcmp('probe', varargin{ii})
                    obj.probe = varargin{ii+1};
                end
            end

            % Get stream info
            obj.file_list        = getAllFilesInDirectory(obj.data_path);
            obj                  = get_ULAOP_DataObj(obj);

            if isempty(obj.probe)
                obj.probe = Probe(strtrim(obj.DataObj.uop.kul.probe_name.str));
            end
            if isempty(obj.scanSequence)
                obj.scanSequence = setupScanSequence_from_uop(obj);
            end
            obj.meta             = get_meta_data(obj);

            obj.meta.nPackets    = obj.scanSequence.nPackets; % Needed for PW and DW sequences
            obj.scanSequence.PRF = obj.DataObj.uop.ssg.prf.num;
        end

        function [date,time] = get_ulaop_acq_date_time(obj)
            date = [strtrim(obj.DataObj.uop.saveinfo.acqstarttime(3).str),'/',strtrim(obj.DataObj.uop.saveinfo.acqstarttime(2).str),'/',strtrim(obj.DataObj.uop.saveinfo.acqstarttime(1).str)];
            time = [strtrim(obj.DataObj.uop.saveinfo.acqstarttime(4).str),':',strtrim(obj.DataObj.uop.saveinfo.acqstarttime(5).str),':',strtrim(obj.DataObj.uop.saveinfo.acqstarttime(6).str)];
        end

        function meta = get_meta_data(obj)
            meta.c0 = obj.DataObj.uop.workingset.soundspeed.num;
            meta.f0 = eval(['obj.DataObj.uop.item' num2str(obj.slice_idx) '.txsettings.txfreq.num']);
            meta.SamplingFreq = obj.DataObj.fs;
            meta.NrSamples = obj.nGate;
            meta.ch_data_time = obj.DataObj.LastReadTime(1) + (0:obj.nGate-1) / obj.DataObj.fs; %time axis
            meta.Range_Min = obj.DataObj.LastReadTime(1) * meta.c0 / 2;
            meta.Range_Max = meta.ch_data_time(end) * meta.c0 / 2;
            meta.MLTnr = obj.scanSequence.nMLTs;
            meta.NrOfImageLines = obj.scanSequence.nScanLines;
            meta.nTxEvents = length(obj.scanSequence.txEvents);
            meta.valid_FRD_frames = get_nValid_FRD_frames(obj);
        end

        function nValid_FRD_frames = get_nValid_FRD_frames(obj)
            nPri = GetTotalPri(obj.DataObj);
            switch(obj.data_type)
                case "RFPre"
                    nValid_FRD_frames = 1:round(nPri/length(obj.scanSequence.txEvents));
                case "IQPost"
                    nValid_FRD_frames = 1:round(nPri/prod(obj.scanSequence.nScanLines));
                case "RFPost"
                    nValid_FRD_frames = 1:round(nPri/prod(obj.scanSequence.nScanLines)/obj.scanSequence.nPackets);
            end
        end

        function obj = get_ULAOP_DataObj(obj)
            if obj.data_type == "RFPre"
                if obj.file_list.rff256 ~= ""
                    % Pre beamformed RF data
                    for idx = 1:length(obj.file_list.rff256)
                        rff256 = DataUlaop256PreBeamforming(obj.file_list.rff256{idx},'itemN',0);
                        if rff256.uos.info.slice.num == obj.slice_idx
                            obj.DataObj = rff256;
                        end
                    end
                elseif obj.file_list.rfm ~= ""
                    % Pre beamformed RF data from BIG memory board
                    for idx = 1:length(obj.file_list.rfm)
                        rfm = DataUlaopPreBeamformingBigMem(obj.file_list.rfm{idx},'itemN',0);
                        if rfm.uos.info.slice.num == obj.slice_idx
                            obj.DataObj = rfm;
                        end
                    end
                end
                obj.nGate = eval(['obj.DataObj.uop.acquirerf_pre.slice' num2str(obj.slice_idx) '.num']);
            elseif obj.data_type == "IQPost" & obj.file_list.uob ~= ""
                % Post beamformed IQ data
                for idx = 1:length(obj.file_list.uob)
                    uob = DataUlaopBaseBand(obj.file_list.uob{idx},'itemN',0);
                    if uob.uos.info.slice.num == obj.slice_idx
                        obj.DataObj = uob;
                    end
                end
                obj.nGate = eval(['obj.DataObj.uop.acquireiq.slice' num2str(obj.slice_idx) '.num']);
            elseif obj.data_type == "RFPost" & obj.file_list.rfb256 ~= ""
                % Post beamformed RF data
                nBeamformers = 1;
                for idx = 1:length(obj.file_list.rfb256)
                    rfb256 = DataUlaop256PostBeamforming(obj.file_list.rfb256{idx}, 'nBeamformer', nBeamformers,'itemN',0);
                    if rfb256.uos.info.slice.num == obj.slice_idx
                         if length(eval(['rfb256.uop.sequencer.item' num2str(obj.slice_idx)])) > 1
                            nBeamformers = eval(['rfb256.uop.sequencer.item' num2str(obj.slice_idx) '(3).num']);
                            obj.DataObj = DataUlaop256PostBeamforming(obj.file_list.rfb256{idx}, 'nBeamformer', nBeamformers,'itemN',0);
                         else
                            obj.DataObj = rfb256;
                         end
                    end
                end
                obj.nGate = eval(['obj.DataObj.uop.acquirerf_post.slice' num2str(obj.slice_idx) '.num']);
            else
                error('Unrecognised data_type must be either: "RFPre", "RFPost" or "IQPost"')
            end
            % We read one pri just to get the "LastReadTime" variable
            Read(obj.DataObj,'firstPri',1,'npri',1);
        end

        function scanSequence = setupScanSequence_from_uop(obj)
            kul = obj.DataObj.uop.kul;
            CompoundingMode = kul.compoundingmode.num;
            DWSlidingTx = kul.dwslidingtx.num;
            txFocusDepth  = kul.txfocusdepth.num;
            nTxEvents = [kul.ntxevents(1).num,kul.ntxevents(2).num];
            nMLT = [kul.nmlt(1).num,kul.nmlt(2).num];
            nLines = [kul.nlines(1).num,kul.nlines(2).num];
            opening_angle = [kul.opening_angle(1).num,kul.opening_angle(2).num];
            scanSequence = scanSequencer(txFocusDepth , nTxEvents(1), nTxEvents(2), opening_angle, nLines, CompoundingMode, DWSlidingTx, obj.probe);
            if prod(nMLT) > 1
                scanSequence = makeSequenceMLT('regular',scanSequence,nMLTs);
            end
        end
    end
end