function [BfMeta, PostMeta]=setUpReconstructionStructure(ConfigFileData)
%%
%  [RECONSTRUCTION]=setUpReconstructionStructure(CONFIGFILEDATA)
%  INPUT: CONFIGFILEDATA is the configuration file encoded into a data
%              structure, this is acquired by running
%              ReadConfigurationFile.m
%  OUTPUT: RECONSTRUCTION is the structure where there are encoded all the
%                  reconstruction parameters i.e image lines, receive
%                  apodization, filtering of the channel data, horizonal
%                  and vertical range, size of the image in pixels.
%  Vangjush Komini,(KU Leuven, 2016)

%%
BfMeta.imgRanges=linspace(ConfigFileData.Range_Min,ConfigFileData.Range_Max,ConfigFileData.NrSamples);
BfMeta.nScanLines=[ConfigFileData.NrOfImageLines,1];
BfMeta.rxApex=[0 0 0];
BfMeta.rxApod = aperture('window', 'rect');
BfMeta.expandApert=0;
PostMeta.outImgSize=[1024 1024];
PostMeta.filterBmode =1;
PostMeta.max_Bmode_range = ConfigFileData.Range_Max - .005; % remove the last 5mm to avoid strong artifacts created by the NaNs on the Beamformer

try
    if strcmp(ConfigFileData.ScanMode,'Anatomical')
        PostMeta.MaxAngle=ConfigFileData.MaxAngle;
        PostMeta.ScanConvMode='Anatomical';
    else
        PostMeta.ScanConvMode='NA';
    end


switch(ConfigFileData.ScanMode)
    case 'MLT'
        try
            BfMeta.MLTnr=ConfigFileData.MLTnr;
        catch
            BfMeta.MLTnr = 1;
        end
end
end
% BfMeta.PhaseCohImg   = {'SCF(0.4)'};
   
end