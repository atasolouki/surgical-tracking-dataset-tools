function BfDataset = readBeamformedDataULAOP(obj,frame_idx)

BfMeta = obj.meta.Beamforming;
CfgMeta = obj.meta.Unpacking;
ULAOP_DataObj = obj.meta.ULAOP_DataObj;
Sequence = obj.meta.Sequence;

if ~isfield(BfMeta,'bf_angles')
    BfMeta.bf_angles  = unique([obj.meta.Sequence.txEvents(:).rxAngles]', 'rows')'; % Compounding had redundant rxAngles
end
BfMeta.nScanLines = obj.meta.Sequence.nScanLines;

if obj.meta.data_type == "IQPost" && obj.meta.file_list.uob ~= ""
    % Post beamformed IQ data
    beginIndex = (frame_idx-1)*prod(CfgMeta.NrOfImageLines) + 1;
    Read(ULAOP_DataObj,'firstPri',beginIndex,'npri',prod(CfgMeta.NrOfImageLines)); % Read all the avaialble PRIs
    [nGate,~] = size(ULAOP_DataObj.LastReadData);
    SetOffsetPri(ULAOP_DataObj,prod(CfgMeta.NrOfImageLines));
    BfData.BfData = ULAOP_DataObj.LastReadData; 
    
elseif obj.meta.data_type == "RFPost" && obj.meta.file_list.rfb256 ~= ""
    % Post beamformed RF data
    SetOffsetPri(ULAOP_DataObj,prod(CfgMeta.NrOfImageLines)); % We offset by nLines because ULA-OP considers each precompounded packet to be one image
    beginIndex = (frame_idx-1)*prod(CfgMeta.NrOfImageLines)*CfgMeta.nPackets + 1;
    Read(ULAOP_DataObj,'firstPri',beginIndex,'npri',prod(CfgMeta.NrOfImageLines)*CfgMeta.nPackets); % Read all the avaialble PRIs
    [nGate,nPRI] = size(ULAOP_DataObj.LastReadData);
    rawData = double(ULAOP_DataObj.LastReadData);

    if Sequence.compoundingMode == 0
        BfData.BfData = reshape(rawData, nGate, CfgMeta.NrOfImageLines(1),CfgMeta.NrOfImageLines(2), 1, []);
    else
        % [nDepths,nChannelsAz,nChannelsEl,1,1,nPackets]
        BfData.BfData = zeros(nGate,CfgMeta.NrOfImageLines(1),CfgMeta.NrOfImageLines(2),CfgMeta.nPackets); %not sure if works for volume dataset
        rawData = reshape(rawData,[nGate,nPRI/CfgMeta.nPackets,CfgMeta.nPackets]);
        for packetID = 1:CfgMeta.nPackets
            BfData.BfData(:,:,:,packetID) = rawData(:,:, packetID);
        end
    end
end

BfMeta.imgRanges = (ULAOP_DataObj.LastReadTime(1)+(0:nGate-1)/ULAOP_DataObj.fs)*CfgMeta.c0/2;

%packing the dataset: same for IQ and postBF data
BfDataset = packBfDataset(obj, BfData, BfMeta);
end
