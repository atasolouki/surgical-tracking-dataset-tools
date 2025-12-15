classdef aperture
    % APERTURE deals with beam apodization (static or dynamic)
    %
    % Pedro Santos (pedro.santos@kuleuven.be)
    %
    % Last update: 16/11/2018
    
    %% properties
    properties  (SetAccess = public)
        origin       = [0 0 0];          % Location of the aperture center in space
        window       = 'rectwin';        % String defining the apodization window type and parameter (e.g., 'Hamming', 'Gauss(8)', 'Tukey(0.5)')
        f_number     = [2 2]             % Desired F-number of the aperture [Az, El]
        fixed_size   = [0 0]             % if >0, it overwrites the dynamic aperture given by the f_number [m, m]
        minimum_size = [0, 0]            % if >0, it sets a minimum for the dynamic aperture given by the f_number [m, m]
    end
    
    
    %% methods
    methods (Access = public)
        
        % Constructor
        function h = aperture(varargin)
            eval(['mco = ?' class(h) ';']);
            plist = mco.PropertyList;
            % varagin
            for n=1:2:(2*floor(nargin/2))
                found=false;
                for m=1:length(plist)
                    if strcmp(plist(m).Name,varargin{n})
                        h.(plist(m).Name)=varargin{n+1};
                        found=true;
                        continue;
                    end
                end
                if ~found, warning('Parameter %s not in %s',varargin{n},class(h)); end
            end
        end
        
        
        function apod_matrix = computeApodization(h, probe, ranges, tilt)
            %computeApodization calculates apodization for a given beam
            %  [APOD] = computeApodization(PROBE, RANGES).
            %  Inputs:  RANGES: ranges where apodization should be computed (for expanding aperture)
            %  Returns: APOD array of size [nRanges, nElementsAz, nElementsEl] with apodization values (nRanges = 1 if no dynamic aperture is used)
            %
            % See also WINDOW
            
            if ~isequal(h.origin, [0 0 0]), keyboard, end % not tested yet
            
            % Compute geometric aperture
            [activeAperture] = getActiveAperture(h, probe, ranges, tilt);
            
            % Find active elements
            active_az = bsxfun(@le, abs(probe.elemenPos(:,1,1) - h.origin(1)), activeAperture(1,:));
            active_el = bsxfun(@le, abs(probe.elemenPos(1,:,2)' - h.origin(2)), activeAperture(1,:));
            
            % Compute apodization weights only when aperture changes
            apod_matrix = zeros(size(active_az, 2), probe.nElementsAz, probe.nElementsEl);
            [~, recalc_idx, out_idx] = unique([active_az; active_el]', 'rows');
            for idx = 1:numel(recalc_idx)
                target_idx = recalc_idx(idx);
                local_apod = computeApodizationStatic(h, nnz(active_az(:,target_idx)), nnz(active_el(:,target_idx)));
                apod_matrix(out_idx==idx, active_az(:,target_idx), active_el(:,target_idx)) = repmat(local_apod, nnz(out_idx==idx), 1, 1);
            end
            
        end
    end
    
    
    %% Provate methods
    methods (Access = private)
        
        function [activeAperture] = getActiveAperture(h, probe, ranges, tilt)
            % Compute the active aperture from the F-number
            if all(h.fixed_size == 0)
                activeAperture = bsxfun(@rdivide, ranges([1 1],:), h.f_number' .* cosd(tilt));
                activeAperture = bsxfun(@max, activeAperture, h.minimum_size');
            else
                % Or assume fixed_size
                activeAperture = h.fixed_size';
            end
            % Crop to transducer size
            activeAperture = bsxfun(@min, activeAperture, [probe.xdcSizeAz probe.xdcSizeEl]');
        end
        
        
        function apod_matrix = computeApodizationStatic(h, n_activeAz, n_activeEl)
            
            apod_param = h.extractParams; % Get window type and (optional) parameters
            
            if ischar(apod_param.method)
                switch(apod_param.method)
                    case 'rect'
                        apod_matrix_az = ones(n_activeAz, 1);
                        apod_matrix_el = ones(1, n_activeEl);
                        apod_matrix    = bsxfun(@times, apod_matrix_az, apod_matrix_el);  % [nAz nEl]
                    case'hann'
                        apod_matrix_az = hann(n_activeAz);
                        apod_matrix_el = hann(n_activeEl)';
                        apod_matrix    = bsxfun(@times, apod_matrix_az, apod_matrix_el);  % [nAz nEl]
                    case {'tukey_offset', 'tukey'}
                        offset = Utils.iif(strcmpi(apod_param.method, 'tukey_offset'), 0.5, 0); % offset 0.08 (hamming) to avoid zero at edges
                        param  = Utils.iif(~isnan(apod_param.value), apod_param.value, .7);
                        apod_matrix_az = offset + tukeywin(n_activeAz, param);
                        apod_matrix_el = offset + tukeywin(n_activeEl, param)';
                        apod_matrix    = bsxfun(@times, apod_matrix_az, apod_matrix_el);  % [nAz nEl]
                        apod_matrix    = apod_matrix / max(apod_matrix(:));
                    otherwise
                        % try apodizations recognized by window() function
                        if h.window(1) ~= '@', apod_param.method = strcat('@', apod_param.method); end
                        try
                            eval(sprintf('apod_matrix_az = window(%s,n_activeAz);', apod_param.method));
                            eval(sprintf('apod_matrix_el = window(%s,n_activeEl)'';', apod_param.method));
                            apod_matrix    = bsxfun(@times, apod_matrix_az, apod_matrix_el);  % [nAz nEl]
                            apod_matrix    = apod_matrix / max(apod_matrix(:));
                        catch
                            error('%s apodization not implemented!', apod_param.method);
                        end
                end
                apod_matrix    = shiftdim(apod_matrix, -1); % to be changed when expanding Ap is implemented
                
            else
                % Input was the apodization matrix, just check for size consistency
                if size(h.window, 2) == n_activeAz && size(h.window, 1) == n_activeEl
                    apod_matrix = h.window;
                else
                    error('Receive apodization wrongly set. Input must be either a string or a matrix with dimensionsn_crystalsEl x n_crystalsAz!')
                end
            end
        end
        
        
        function [params] = extractParams(h)
            % EXTRACTPARAMS extracts the apodization name and parameter value.
            % PARAMS = aperture.extractParams returns a structure with 'window', 'value'
            
            str_to_match = '(?<method>[a-zA-Z]+)\(?(?<value>(\d*\.?\d*))\)?'; % Look for method and then valued enclosed in parentheses (1 values accepted)
%             str_to_match = '(?<method>[a-zA-Z]+)\(?(?<val1>(\d*\.?\d*)),?\/?(?<val2>(\d*\.?\d*))\)?'; % Look for method and then valued enclosed in parentheses (2 values are accepted and must be slipt by either a coma or a forward slash
            params = regexp(h.window, str_to_match,'names');
            params.value = str2double(params.value);

        end % extractParams
        
        
    end
    
    %% set methods
    methods
        function h=set.f_number(h,in_aperture)
            h.f_number = h.assert_dimensions(in_aperture);
        end
        function h=set.fixed_size(h,in_aperture)
            h.fixed_size = h.assert_dimensions(in_aperture);
        end
        function h=set.minimum_size(h,in_aperture)
            h.minimum_size = h.assert_dimensions(in_aperture);
        end
        function in_aperture = assert_dimensions(h,in_aperture)
            assert(isa(in_aperture,'single')||isa(in_aperture,'double'), 'The input should be a single or double');
            assert(numel(in_aperture)>0 && numel(in_aperture)<=2 , 'The fixed_aperture must be 1 or two elements [fixed_aperture_ x fixed_aperture_y]');
            if numel(in_aperture)==1
                in_aperture = [in_aperture 0];
            end
        end
    end
    
end

