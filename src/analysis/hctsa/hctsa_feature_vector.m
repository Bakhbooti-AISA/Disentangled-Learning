function [feat_vec, feat_names] = hctsa_feature_vector(ts, mode)
%HCTSA_FEATURE_VECTOR Compute features for a single univariate time series.
%
% Inputs
%   ts   : (T x 1) or (1 x T) numeric vector
%   mode : 'full' (default) or 'catch22'
%
% Outputs
%   feat_vec   : (1 x F) double
%   feat_names : (1 x F) cell array of char

    if nargin < 2 || isempty(mode)
        mode = 'full';
    end

    % ---- sanitize input ----
    if isempty(ts)
        feat_vec = [];
        feat_names = {};
        return;
    end

    ts = double(ts(:));      % force column vector double
    ts = ts(isfinite(ts));   % drop NaN/Inf

    if numel(ts) < 10
        feat_vec = [];
        feat_names = {};
        return;
    end

    mode = lower(string(mode));

    switch mode
        case "full"
            % Full hctsa feature library (can be slow).
            % Note: Different hctsa versions differ in outputs; try to get names.
            try
                [feat_vec, feat_names] = TS_CalculateFeatureVector(ts, 0);
            catch
                feat_vec = TS_CalculateFeatureVector(ts, 0);
                feat_names = arrayfun(@(i) sprintf('feat_%d', i), 1:numel(feat_vec), 'UniformOutput', false);
            end

        case "catch22"
            % catch22 is NOT guaranteed to be included in hctsa installs.
            % If you have a catch22 MATLAB implementation on your path,
            % this will work. Otherwise, we raise a clear error.
            %
            % Common function names in MATLAB ports vary; adjust if needed.
            if exist('catch22_all', 'file') == 2
                % Many ports expose: [values, names] = catch22_all(ts);
                [vals, names] = catch22_all(ts);
                feat_vec = double(vals(:))';
                feat_names = cellstr(names(:))';
            elseif exist('CO_catch22', 'file') == 2
                % Some toolboxes expose CO_catch22(ts) returning a struct
                out = CO_catch22(ts);
                feat_vec = double(out.values(:))';
                feat_names = cellstr(out.names(:))';
            else
                error(['mode="catch22" requested, but no catch22 function found on MATLAB path. ' ...
                       'Add a catch22 MATLAB implementation (e.g., catch22_all) or switch to mode="full".']);
            end

        otherwise
            error('Unknown mode "%s". Use "full" or "catch22".', mode);
    end

    % Ensure row outputs
    feat_vec = double(feat_vec(:))';
    if isstring(feat_names)
        feat_names = cellstr(feat_names);
    end
    feat_names = feat_names(:)';

end
