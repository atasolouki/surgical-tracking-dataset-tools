function acquisition_names = map_multi_acquisition_files(main_path)

% Get a list of all files in the folder
uop_files = dir(fullfile(main_path, '*.uop'));

all_files = dir(fullfile(main_path, '*.*'));
all_files = all_files(~ismember({all_files.name}, {'.', '..'}));
acquisitionMap = containers.Map;

if isempty(uop_files)
    folder_names = all_files([all_files.isdir]);
    for folder_idx = 1:length(folder_names)
        acquisitionMap(folder_names(folder_idx).name) = {};
    end
    acquisition_names = keys(acquisitionMap);
    return
end
all_files = all_files(~[all_files.isdir]);

% acquisitionMap = containers.Map;

for uop_file_idx = 1:length(uop_files)
    currentFile = uop_files(uop_file_idx).name;

    acquisitionName = strrep(currentFile,'.uop', '');
    acquisitionName = strrep(acquisitionName,'_params' ,'');

    if contains(acquisitionName,'CypID')
        acquisitionName = strrep(acquisitionName,'CypID0_', '');
        acquisitionName = strrep(acquisitionName,'CypID1_', '');
    end

    % If the acquisition name is empty, skip the file
    if isempty(acquisitionName)
        continue;
    end
    acquisitionMap(acquisitionName) = {};

end

acquisition_names = keys(acquisitionMap);

for file_idx = 1:length(all_files)
    currentFile = all_files(file_idx).name;
    [~, currentFile, ~] = fileparts(currentFile);

    if contains(currentFile,'CypID')
        currentFile = strrep(currentFile,'CypID0_', '');
        currentFile = strrep(currentFile,'CypID1_', '');
    end

    for acquisition_idx = 1:length(acquisition_names)
        acquisitionName = acquisition_names{acquisition_idx};
        if startsWith(currentFile, acquisitionName)
            this_map = acquisitionMap(acquisitionName);
            this_map{end+1,1} = all_files(file_idx).name;
            acquisitionMap(acquisitionName) = this_map;
        end
    end
end

for idx = 1:length(acquisition_names)
    acquisitionName = acquisition_names{idx};
    subFolder = fullfile(main_path, acquisitionName);
    if ~exist(subFolder, 'dir')
        mkdir(subFolder);
    end
    filesToMove = acquisitionMap(acquisitionName);
    if sum(contains(filesToMove,'.uop')) > 1
        subFolder0 = fullfile(subFolder, 'CypID0');
        if ~exist(subFolder0, 'dir'), mkdir(subFolder0); end
        subFolder1 = fullfile(subFolder, 'CypID1');
        if ~exist(subFolder1, 'dir'), mkdir(subFolder1); end
    end

    for jdx = 1:length(filesToMove)
        if contains(filesToMove{jdx},'CypID0') == 1
            try
                movefile(fullfile(main_path, filesToMove{jdx}), fullfile(subFolder0, filesToMove{jdx}));
            catch
                movefile(fullfile(main_path, filesToMove{jdx}), fullfile(subFolder, filesToMove{jdx}));
            end
        elseif contains(filesToMove{jdx},'CypID1') == 1
            try 
                movefile(fullfile(main_path, filesToMove{jdx}), fullfile(subFolder1, filesToMove{jdx}));
            catch
                movefile(fullfile(main_path, filesToMove{jdx}), fullfile(subFolder, filesToMove{jdx}));
            end
        else
            movefile(fullfile(main_path, filesToMove{jdx}), fullfile(subFolder, filesToMove{jdx}));
        end
    end
end
end







