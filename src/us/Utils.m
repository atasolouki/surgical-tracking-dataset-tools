classdef Utils
    % General utilities.
    %
    % compareStructures    - compares the fields (and values) of two objects
    % iif                  - ternary operator
    % beam2cart            - convert azimuth/elevation angles to cartesian coordinates
    % dirSubFolders        - Lists all subfolders in a given path except '.' and '..'
    % figureMax            - create new figure maximized to screen size
    % numSubplots          - calculates ideal nr of rows/columns for N subplots    % parsave           - save data inside a parfor loop
    % parsave              - save data inside a parfor loop
    % rotx                 - rotation matrix around x-axis
    % roty                 - rotation matrix around y-axis
    % rotz                 - rotation matrix around z-axis
    % getNrOfFrameFiles    - get a list of the frame files in a given dir
    % splitInSubarrays     - split a larger array size into smaller subsets
    % getParamFromFile     - reads a given parameter from a text file and returns its value
    % getParamFromString   - get a parameter value from a text file
    % getCurrentFunction   - return the name of the function and file being executed
    % error_warning_msg    - display error or waring message
    % getSizeOfBinaryArray - return size of next array in Labview binary file
    % maxn                 - computes the global maximum of an N-D array
    
    
    methods (Static)
        
        
        function out = iif(cond, val_true, val_false)
            % IIF implements a ternary operator.
            % OUT = Utils.iif(CONF, VAL_TRUE, VAL_FALSE) returns VAL_TRUE if the
            % condition is true, or VAL_FALSE if condition is false
            if cond,
                out = val_true;
            else
                out = val_false;
            end
        end % iif
        
        
        function [x, y, z] = beam2cart(az_angle, el_angle, range)
            % BEAM2CART converts azimuth/elevation angles to cartesian coordina tes
            % [X, Y, Z] = Utils.beam2cart(AZ, EL, R) takes as input:
            % Inputs: azimuth (AZ) and elevation (EL) angles (measured from range axis z) in degrees, and the range R.
            % It returns the cartesian coordinates X (azimuth), Y (elevation) and Z (range).
            
            if nargin < 3,
                range = 1;
            end
            
            % Degrees to radians (faster)
            az_angle = az_angle .* (pi/180);
            el_angle = el_angle .* (pi/180);
            
            
            % Works similarly to sph2cart. But x/z and y/x axis are swapped
            x = range .* cos(el_angle) .* sin(az_angle);
            y = range .* sin(el_angle);
            z = range .* cos(el_angle) .* cos(az_angle);
            
            if nargout < 2,
                x = [x y z];  % concatenate all coordinates
            end
            
        end % beam2cart
        
        
        function [az_angle, el_angle, range] = cart2beam(x, y, z)
            % CART2BEAM converts cartesial coordinates to azimuth/elevation angles
            % [AZ_ANGLE, EL_ANGLE] = Utils.cart2beam(X, Y, Z) takes as input:
            % Inputs: azimuth (x), elevation (y) and range (z) coordinates.
            % It returns the beam angles in azimuth (AZ_ANGLE) and elevation (EL_ANGLE)and the range (Z).
            
            if nargin < 3,
                range = 1;
            end
            
            
            % Works similarly to car2sph. But x/z and y/x axis are swapped
            az_angle = atan2(x,z);
            el_angle = atan2(y,sqrt(z.^2 + x.^2));
            range = sqrt(x.^2 + y.^2 + z.^2);
            
            
            % Degrees to degrees
            az_angle = az_angle .* (180/pi);
            el_angle = el_angle .* (180/pi);
            
            
            if nargout == 1,
                az_angle = [az_angle el_angle];  % concatenate all coordinates
            end
            
        end % cart2beam
        
        
        
        
        function [p,n]=numSubplots(n)
            % NUMSUBPLOTS calculates how many rows/columns of subplots are needed to neatly display N subplots.
            % [P, Q] = numSubplots(N)
            % inputs: N - the desired number of subplots.
            % Outputs: P - a vector length 2 defining the number of rows and columns required to show n plots.
            % Q - (optional) the current number of subplots. This output is used only by this function for a recursive call.
            % © Rob Campbell - January 2010
            
            while isprime(n) & n>4,
                n=n+1;
            end
            
            p=factor(n);
            
            if length(p)==1
                p=[1,p];
                return
            end
            
            while length(p)>2
                if length(p)>=4
                    p(1)=p(1)*p(end-1);
                    p(2)=p(2)*p(end);
                    p(end-1:end)=[];
                else
                    p(1)=p(1)*p(2);
                    p(2)=[];
                end
                p=sort(p);
            end
            
            %Reformat if the column/row ratio is too large: we want a roughly square design
            while p(2)/p(1)>2.5
                N=n+1;
                [p,n]=Utils.numSubplots(N); %Recursive!
            end
        end  % numSubplots
        
        
        function fig_in = figureMax(fig_in)
            if nargin < 1
                fig_in = figure;
            else
                figure(fig_in)
            end
            
            set(gcf,'units','normalized','outerposition',[0 0 1 1])
        end
        
        function parsave (savefile,varargin)
            % PARSAVE allows saving variables to a .mat-file while in a parfor loop.
            % parsave(FileName,Variable1,Variable2,...)
            % Note: do NOT pass the variable names but instead the variable itself, e.g. parsave('file.mat',x,y);
            % © Joost H. Weijs - 2016
            
            for i=1:nargin-1
                %Get name of variable
                name{i}=inputname(i+1);
                
                %Create variable in function scope
                eval([name{i} '=varargin{' num2str(i) '};']);
            end
            
            %Save all the variables, do this by constructing the appropriate command
            %and then use eval to run it.
            comstring=['save(''' savefile ''''];
            for i=1:nargin-1
                comstring=[comstring ',''' name{i} ''''];
            end
            comstring=[comstring ');'];
            eval(comstring);
        end
        
        
        function R = rotx(phi)
            %ROTX  rotate around X by PHI
            %	R = ROTX(PHI)
            % © Brad Kratochvil - 2005
            
            R = [1        0         0; ...
                0 cos(phi) -sin(phi); ...
                0 sin(phi)  cos(phi)];
            
            % this just cleans up little floating point errors around 0
            if exist('roundn'),
                R = roundn(R, -15);
            end
        end
        
        
        function R = roty(beta)
            %ROTY  rotate around Y by BETA
            %	R = ROTY(BETA)
            % © Brad Kratochvil - 2005
            
            R = [cos(beta) 0 sin(beta); ...
                0 1         0; ...
                -sin(beta) 0 cos(beta)];
            
            
            % this just cleans up little floating point errors around 0
            % so that things look nicer in the display
            if exist('roundn'),
                R = roundn(R, -15);
            end
            
        end
        
        
        function R = rotz(alpha)
            %ROTZ  rotate around Z by ALPHA
            %	R = ROTZ(ALPHA)
            % © Brad Kratochvil - 2005
            
            R = [cos(alpha) -sin(alpha) 0; ...
                sin(alpha)  cos(alpha) 0; ...
                0           0 1];
            
            % this just cleans up little floating point errors around 0
            % so that things look nicer in the display
            if exist('roundn'),
                R = roundn(R, -15);
            end
            
        end
        
        
        
        
        function frame_nr = getNrOfFrameFiles(pathDir, fileName)
            AllFilesDir  = dir(fullfile(pathDir, fileName));
            if ~isempty(AllFilesDir)
                AllFilesDir = struct2cell(AllFilesDir);
                AllFilesDir = AllFilesDir(1,:);
                frame_nr    = regexp(AllFilesDir, 'frame(\d+)', 'tokens', 'once');
                n_digits = max(cellfun(@(x) numel(x{1}), frame_nr)); % Convert all strings to the largest number of digits (initial assumption that all frames have 5 digits no longer applies)
                frame_nr = cellfun(@(x){[repmat('0', [1 n_digits-numel(x{1})]) x{1}]}, frame_nr, 'UniformOutput', 0);
                frame_nr    = sort(str2num(cell2mat([frame_nr{:}]')));
            else
                frame_nr = [];
            end
        end
        
        
        function data_type = getAvailableDataType(pathDir, fileName)
            try
                file_struct = matfile(fullfile(pathDir, fileName));
                data_type = fieldnames(file_struct);
                data_type = setdiff(data_type, {'Properties', 'meta'});
            catch
                data_type = '';
            end
        end
        
        function selected_type = selectDataFromAvailableTypes(pathDir, fileName, select_mode)
            % select_mode: 'single' or 'multiple'
            
            if ~exist('select_mode', 'var') || isempty(select_mode)
                select_mode = 'single';
            end
            
            % Check which data types exist (TODO: combine all possible types, including ChData and TVI)
            data_types = Utils.getAvailableDataType(pathDir, fileName);
            
            % Select one from the available types
            if numel(data_types) > 1
                [answr_idx, answr_ok] = listdlg('PromptString','Select a data type:', 'SelectionMode',select_mode, 'ListString',data_types);
                if answr_ok
                    selected_type = data_types(answr_idx);
                else
                    return
                end
            elseif numel(data_types) == 1
                selected_type(1) = data_types;
            else
                fprintf('\tERROR: No data found in the desired file %s.\n', fullfile(pathDir, fileName))
                selected_type = '';
            end
            
            
            
        end
        
        
        function array_subset = splitInSubarrays(array_in, array_length)
            
            if rem(numel(array_in), array_length) % last subset is incomplete
                n_full_blocks = floor(numel(array_in)/array_length);
                array_subset = num2cell(reshape(array_in(1:n_full_blocks*array_length), array_length, n_full_blocks), 1);
                array_subset{end+1} = array_in(n_full_blocks*array_length+1:end)';
            else
                array_subset = num2cell(reshape(array_in, array_length, []), 1);
            end
            
            
        end
        
        
        
        
        function [value] = getParamFromFile(FileName, paramName, precision)
            % Reads a given parameter from a text file and returns its value.
            % VALUE = GETPARAMFROMFILE(FULL_PATH, PAR_NAME, PRECISION)
            
            % Pedro Santos (KU Leuven, 2017)


            
            if isstruct(FileName)
                % If we have a dir file, contruct the full path
                FileName = fullfile(FileName.folder, FileName.name);
            end
            if ~exist('precision', 'var'), precision = ''; end
            
            
            % Read file
            fileID = fopen(FileName);
            read_str = textscan(fileID, '%s','delimiter', '\n');
            fclose(fileID);
            
            
            % Read parameter from string
            value = Utils.getParamFromString(read_str{1}, paramName, precision);
            
        end
        
        
        function [value] = getParamFromString(fileStr, paramName, precision, default_val)
            % Reads a given parameter from a text string and returns its value
            %
            % [value] = getParamFromString(fileStr, paramName, precision, default_val)
            %
            % Pedro Santos (KU Leuven, last update: 23/01/2019)
            
            % look for the variable and extract value
            fileStr = regexprep(fileStr, '\s*=\s*', '=');       % remove spaces around =
            fileStr = regexprep(fileStr, '\s$', '');  % remove spaces at the end
            fileStr = regexprep(fileStr, '\s*\(\D+\)\s*', '');  % remove units
            fileStr = regexprep(fileStr, ',', '.');             % replace commas by periods
            Index = regexp(fileStr, sprintf('^%s=*', paramName)); % use ^ and = to make sure we don't return a parameter that matches partially
            if ~isempty(Index) && ~isempty([Index{:}])
                Index = find(~cellfun(@isempty, Index));
                value = fileStr{Index(1)}(length(paramName)+2:end);  % look for numeric after the paramName (add +2 because we need to skip the =)
                if ~exist('precision', 'var') || isempty(precision) || ~strcmpi(precision, 'string')
                    value = str2num(value); % This works for numeric arrays but not for strings (which return NaNs)
                end
            else
                if exist('default_val', 'var') && ~isempty(default_val)
                    value = default_val;
                else
                    value = [];
                end
                fprintf('\t@getParamFromString: The parameter "%s" was not found in the given string. \n', paramName)
            end
        end
        
        function dir_clean = dirSubFolders(varargin)
            % Lists all subfolders in a given path except
            
            if nargin == 0
                path_name = '.';
            elseif nargin == 1
                path_name = varargin{1};
            else
                error('Too many input arguments.')
            end
            
            
            dir_all = dir(path_name);      % Call Matlab DIR
            dir_folders = dir_all([dir_all.isdir]);  % keep only folders
            dir_clean = dir_folders(~ismember({dir_folders.name},{'.','..'}));     % Remove '.' and '..'
            
        end
        
        
        
        function [function_name, file_name, line_number] = getCurrentFunction(inarg, ignore_level)
            % getCurrentFunction - return the name of the function and file being executed
            
            if ~exist('ignore_level', 'var') || isempty('ignore_level')
                ignore_level = 1;
            end
                
            % Get function name
            dbk = dbstack(ignore_level); % ignore N frames, to get rid of e.g. Utils.getCurrentFunction
            if isempty(dbk)
                str = 'base';
                line_number = NaN;
            else
                str = dbk(1).name; % last function to be called
                line_number = dbk(1).line;
            end
            ixf = find( str == '.', 1, 'first');
            if isempty( ixf ) || ( nargin==1 && strcmp( inarg, '-full' ) )
                function_name = str;
                file_name = str;
            else
                function_name = str( ixf+1 : end );
                file_name = str(1:ixf-1);
            end
        end
        
        
        function [] = error_warning_msg(error_msg, level, title_str)
            % error_warning_msg(MSG, LEVEL, TITLE) creates warnings or errors
            % MSG is the error info string
            % LEVEL can be 'warning' (ide display), 'warning_popup' (ide
            %    display and warning popup), 'error' (ide display and error
            %    popup), 'fatal' (ide display, error popup and crash code).
            % TITLE is the optial title for the popup windows.
            %
            % Pedro Santos (11/01/2019)
            
            if ~exist('level', 'var') || isempty(level), level = ''; end

            [f_name, f_filename, f_line] = Utils.getCurrentFunction([], 2);
            display_message = sprintf('%s\n\tfile: %s\n\tfunction: %s\n\tline: %d', error_msg, f_filename, f_name, f_line);
            
            switch(lower(level))
                case {'warn', 'warning', 'w'}
                    fprintf('### WARNING ### \t%s', display_message)
                case {'warn_popup', 'warning_popup'}
                    if exist('title_str', 'var') && ~isempty(title_str)
                        warndlg(display_message, title_str);
                    else
                        warndlg(display_message);
                    end
                    fprintf('### WARNING ### \t%s', display_message)    
                case {'error'}
                    if exist('title_str', 'var') && ~isempty(title_str)
                        errordlg(display_message, title_str);
                    else
                        errordlg(display_message);
                    end
                    fprintf('### ERROR ### \t%s', display_message)
                case {'fatal', 'fatal_error'}
                    if exist('title_str', 'var') && ~isempty(title_str)
                        errordlg(display_message, title_str);
                    else
                        errordlg(display_message);
                    end
                    error('### FATAL ERROR ### \t%s', display_message)
            end
            
%             % Trying to output Matlab's error
%             try
%                 dum_way_to_make_this_crash
%             catch ME
%                 error_report = getReport(ME,'extended');
%                 idx = strfind(error_report, 'Error');
%                 error_report = [error_msg error_report(idx(2)-2:end)];
%                 error(error_report);
%                 errordlg(sprintf('%s\n\n@ %s : %s, line %d', error_msg, f_filename, f_name, f_line));
%             end
            
        end
        
        
            
        function dims = getSizeOfBinaryArray(fid, n_dims)
            % getSizeOfBinaryArray(FILE_ID, N_DIMS) returns the size of the
            % next array in Labview binary file FILE_ID given the dimension
            % of the array N_DIMS.

            siz = [2^32 2^16 2^8 1]';  % Array dimension conversion table
            dims = nan(1,n_dims);      % Pre-allocate space
            for ii = 1:n_dims
                temp = fread(fid,4);
                dims(ii) = sum(siz.*temp);
            end
        end
        
        
        function max_val = maxn(array_in)
            max_val = max(array_in(:));
        end
        function is_equal = compareStructures(obj1, obj2)
            % compareStructures compares the fields (and values) of two
            % objects and prints the differences.
            %
            % Pedro Santos (last updated: 04/01/2019)
            
            is_equal    = 1;
            diff_fields = {};
            
            % Check all fields in Probe1
            fields_pr1 = fieldnames(obj1);
            for field_i = fields_pr1'
                if isfield(obj2, field_i{1}) || isprop(obj2, field_i{1})
                    if isequal(obj1.(field_i{1}), obj2.(field_i{1}))
                        is_equal = is_equal && 1;
                    else
                        is_equal = is_equal && 0;
                        diff_fields(end+1,:) = {field_i{1}, obj1.(field_i{1}), obj2.(field_i{1})}; % Add differences to a list, to report at the end
                    end
                else
                    fprintf(' * Attribute %s not found in object 2!\n', field_i{1})
                end
            end
            
            
            % Check all fields in Probe2
            fields_pr2 = fieldnames(obj2);
            for field_i = fields_pr2'
                if isfield(obj1, field_i{1}) || isprop(obj1, field_i{1})
                    if isequal(obj1.(field_i{1}), obj2.(field_i{1}))
                        is_equal = is_equal && 1;
                    else
                        is_equal = is_equal && 0;
                        if ~ismember(field_i{1}, diff_fields(:,1)) % If this field was still not listed, add it
                            diff_fields(end+1,:) = {field_i{1}, obj1.(field_i{1}), obj2.(field_i{1})};
                        end
                    end
                else
                    fprintf(' * Attribute %s not found in object 1!\n', field_i{1})
                end
            end
            
            
            % Print the differences
            if ~is_equal
                fprintf('** The following fields are different between two object compared:\n'), fprintf('\t%s\n', diff_fields{:,1})
%                 fprintf('\t\tField Name:\t\t\t\t Value Probe1: \t\t\t\t Value Probe1: \n')
%                 for ii = 1:size(diff_fields, 1)
%                     fprintf('\t\t%s \t\t\t %s \t\t\t%s\n', diff_fields{ii, 1},  Utils.iif(ischar(diff_fields{ii, 2}), diff_fields{ii, 2}, num2str(diff_fields{ii, 2})),  Utils.iif(ischar(diff_fields{ii, 3}), diff_fields{ii, 3}, num2str(diff_fields{ii, 3})))
%                 end
            end
        end
    end % methods (Static)
end

