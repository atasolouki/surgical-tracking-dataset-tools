function File_List = getAllFilesInDirectory(dirName)
%Get the list of the file at a specific direcotry name
%  FILE_LIST = getAllFilesInDirectory(NAME_OF_DIRECTORY)
%  Inputs: NAME_OF_DIRECTORY: The full path of the directory where the listing of the files in needed.
%
%  Returns: FILE_LIST: Is the a structure where each entity encodes the
%                                  channel data, configuration file, firing angles, channel data into text
%                                  file, into a list of cells
%
%  Example: fileList = getAllFilesInDirectory('C:\Users\vkomin1\Google Drive\Work');
%  Vangjush Komini,(KU Leuven, 2016)
%%

% Get all files in DIR
dirData = dir(dirName);
FileNames = {dirData(~[dirData.isdir]).name}';  % Get a list of the files (skip directories)
if ~isempty(FileNames)
    FileNames = cellfun(@(x) fullfile(dirName,x), FileNames,'UniformOutput',false);
end
    
    
% Predefine struct
File_List.ChannelDataTx        = {};
File_List.ChannelData          = {}; % What is this?
File_List.ConfigFile           = {};
File_List.ConfigFileCustomMode = {};
File_List.TimeStamps           = {};
File_List.Ecg                  = {};
File_List.FiringAngles         = {};
File_List.Datasets             = {};
File_List.orb                  = {};
% ULA-OP Files Below
File_List.uop                  = {};
File_List.rff256               = {};
File_List.rfb256               = {};
File_List.uob                  = {};
File_List.rfm                  = {};
File_List.uos                  = {};

% Now, Loop through all files
for i=1:length(FileNames)
    [~, ~, f_ext] = fileparts(FileNames{i});
    
   
    switch(f_ext)
        case {'.png', '.bstrm', '.fstrm'}
            File_List.ChannelDataTx{end+1} = FileNames{i};
        case '.mat'
            if ~isempty(strfind(lower(FileNames{i}),'dataset')), File_List.Datasets{end+1}=FileNames{i}; end
        case '.bin'
            File_List.ChannelData{end+1} = FileNames{i};
        case '.cfg'
            File_List.ConfigFile{end+1} = FileNames{i};
        case '.ccfg'
            if strfind(FileNames{i}, 'AnatScan.ccfg')
%                 fprintf('Info: %s file ignored from %s (@getAllFilesInDirectory)\n', 'AnatScan.ccfg', dirName)
                % do nothing (this is messy...)
            else
                File_List.ConfigFileCustomMode{end+1} = FileNames{i};
            end
        case '.txt' 
            keyboard % Still needed?
%             if ~isempty(strfind(FileNames{i},'Angle')), File_List.FiringAngles{end+1}=FileNames{i}; end
%             if (~isempty(strfind(Files{i},'FRD'))||~isempty(strfind(Files{i},'ChannelData), File_List.ChannelDataTx{end+1} = FileNames{i}, File_List.ChannelDataTx{end+1} = FileNames{i};
        case '.tmst'
            File_List.TimeStamps{end+1} = FileNames{i};
        case '.becg'
            File_List.Ecg{end+1} = FileNames{i};
        case '.orb'
            File_List.orb{end+1} = FileNames{i};
        case '.uop'
            File_List.uop{end+1} = FileNames{i};
        case '.rff256'
            File_List.rff256{end+1} = FileNames{i};
        case '.rfb256'
            File_List.rfb256{end+1} = FileNames{i};
        case '.uob'
            File_List.uob{end+1} = FileNames{i};
        case '.rfm'
            File_List.rfm{end+1} = FileNames{i};
        case '.uos'
            File_List.uos{end+1} = FileNames{i};
        otherwise
            fprintf('Info: %s file ignored from %s (@getAllFilesInDirectory)\n', f_ext, dirName)
    end  
end

% Make sure we don't allow multiple files for sensitive information
File_List.ConfigFile = selectSingleFile(File_List.ConfigFile);
File_List.ConfigFileCustomMode = selectSingleFile(File_List.ConfigFileCustomMode);
File_List.TimeStamps = selectSingleFile(File_List.TimeStamps);
File_List.Datasets = selectSingleFile(File_List.Datasets);
File_List.Ecg = selectSingleFile(File_List.Ecg);

end

function [file_list] = selectSingleFile(file_list)
% selectSingleFile allows the user to select the correct file of a given type, when more than one was found

    if numel(file_list) == 1
        file_list = {file_list{1}}; % return first
    elseif numel(file_list) > 1
        [~, ~, f_ext] = fileparts(file_list{1});
        [~, file_name] = cellfun(@(x)fileparts(x), file_list, 'UniformOutput', 0);
        [choice_idx] = listdlg('PromptString', sprintf('Select a %s file:', upper(f_ext)), 'SelectionMode','single', 'ListString', file_name);
        if isempty(choice_idx), error('No %s file selected.', f_ext), end
        ConfigFileName = file_list{choice_idx};
        file_list = {ConfigFileName}; % One acquisition can have only 1 cgf, so hide the others
    end

end
