% recipient.m
% MATLAB test stand-in for recipient.py.
%
% Set preferences before running, see example below.  These need to point
% at a directory where the sender and this program will read and write
% files (network_disk_path), and to a directory where there is some
% sample data if we are using that mode.
%
% Responses are random (0 or 1).  M_RGBTo2DW is loaded and available in
% the workspace if you want to add model-based predictions later.

%{
    % Set preferences before running, e.g.:
    setpref('wppm', 'color_thres_base_dir', '/Users/dhb/Dropbox (Personal)/ShareWithX/ShareWithWPPM/TestCommuniation');
    setpref('wppm', 'network_disk_path',    '/Users/dhb/Dropbox (Personal)/ShareWithX/ShareWithWPPM/TestData');
%}

%% Paths from preferences
color_thres_base_dir = getpref('wppm', 'color_thres_base_dir');
network_disk_path    = getpref('wppm', 'network_disk_path');

%% Load RGB->2DW transformation matrix
cal_tag = 'DELL_02242025_copy';
try
    M_RGBTo2DW = loadMatrix(color_thres_base_dir, ...
        sprintf('M_RGBTo2DW_%s.csv', cal_tag));
    fprintf('Loaded M_RGBTo2DW.\n');
catch
    warning('M_RGBTo2DW_%s.csv not found; random responses will be used.', cal_tag);
    M_RGBTo2DW = []; %#ok<NASGU>
end

%% Get experiment info
dlg_answer = inputdlg( ...
    {'Subject ID:', 'Subject Initials:', 'Session number today:'}, ...
    'Experiment Info', 1, {'', '', ''});
if isempty(dlg_answer)
    error('No experiment info entered.');
end
subject_id    = str2double(dlg_answer{1});
subject_init  = strtrim(dlg_answer{2});
session_today = str2double(dlg_answer{3});

%% Locate subject directory and session file
is_practice = true;
if is_practice
    path_sub = fullfile(network_disk_path, sprintf('sub%d', subject_id), 'practice');
else
    path_sub = fullfile(network_disk_path, sprintf('sub%d', subject_id));
end

pat   = sprintf('sub%d_%s*session*.txt', subject_id, subject_init);
files = dir(fullfile(path_sub, pat));
if isempty(files)
    error('No session file found in %s matching %s', path_sub, pat);
end
[~, idx]       = max([files.datenum]);
file_name      = files(idx).name;
full_file_path = fullfile(path_sub, file_name);

validateSessionFile(full_file_path, subject_init, session_today);
fprintf('Using session file: %s\n', file_name);

%% Communication parameters
retry_delay    = 1/60;   % ~16.7 ms (one 60 Hz frame)
timeout        = 1200;   % 20 minutes
response_delay = 0.1;    % seconds before sending simulated response

%% Step 1: Wait for "Set_Up_to_Communicate"
fprintf('Waiting for initialization command...\n');
t_start = tic;
while true
    if strcmp(getLastWord(full_file_path), 'Set_Up_to_Communicate')
        appendToFile(full_file_path, 'Ready_To_Communicate');
        break;
    end
    if toc(t_start) > timeout
        error('Timeout: did not receive Set_Up_to_Communicate.');
    end
    pause(retry_delay);
end
fprintf('Initialization confirmed.\n');

%% Step 2: Main trial loop
terminate     = false;
trial_counter = 0;
t_start       = tic;

while ~terminate
    last_line = getLastLine(full_file_path);
    last_word = getLastWordFromLine(last_line);

    if strcmp(last_word, 'Done')
        terminate = true;

    elseif strcmp(last_word, 'Break')
        pause(1);
        appendToFile(full_file_path, 'Resume');
        t_start = tic;

    elseif strcmp(last_word, 'Image_Display')
        trial_counter = trial_counter + 1;
        fprintf('Trial #%d...\n', trial_counter);

        [trial_type, ref_rgb, comp_rgb, comp2_rgb] = extractRGBvals(last_line);
        pause(response_delay);

        response = randi([0 1]);   % random; replace with Wishart prediction if needed

        if isempty(comp2_rgb)
            str_comp2 = '';
        else
            str_comp2 = sprintf('Comp2_R%.8f_G%.8f_B%.8f ', ...
                comp2_rgb(1), comp2_rgb(2), comp2_rgb(3));
        end
        msg = sprintf( ...
            '%s Ref_R%.8f_G%.8f_B%.8f Comp_R%.8f_G%.8f_B%.8f %sResp%d Image_Confirmed', ...
            trial_type, ...
            ref_rgb(1),  ref_rgb(2),  ref_rgb(3), ...
            comp_rgb(1), comp_rgb(2), comp_rgb(3), ...
            str_comp2, response);
        appendToFile(full_file_path, msg);
        fprintf('RGB values confirmed (response = %d).\n', response);
        t_start = tic;

    else
        if toc(t_start) > timeout
            error('Timeout: sender did not send RGB values in time.');
        end
        pause(retry_delay);
    end
end

fprintf('Communication complete.\n');

%% ---- Helper functions -------------------------------------------------------

function M = loadMatrix(base_dir, filename)
    files = dir(fullfile(base_dir, '**', filename));
    if isempty(files)
        error('File not found: %s under %s', filename, base_dir);
    end
    M = readmatrix(fullfile(files(1).folder, files(1).name), 'NumHeaderLines', 0);
end

function last_line = getLastLine(file_path)
    fid = fopen(file_path, 'r');
    last_line = '';
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line) && ~isempty(strtrim(line))
            last_line = strtrim(line);
        end
    end
    fclose(fid);
end

function word = getLastWord(file_path)
    word = getLastWordFromLine(getLastLine(file_path));
end

function word = getLastWordFromLine(line)
    parts = strsplit(strtrim(line));
    if isempty(parts) || isempty(parts{1})
        word = '';
    else
        word = parts{end};
    end
end

function appendToFile(file_path, message)
    [~, hostname] = system('hostname');
    hostname  = strtrim(hostname);
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    fid = fopen(file_path, 'a');
    if fid == -1
        error('Cannot open file for appending: %s', file_path);
    end
    fprintf(fid, '%s - %s: %s\n', timestamp, hostname, message);
    fclose(fid);
end

function [trial_type, ref_rgb, comp_rgb, comp2_rgb] = extractRGBvals(line)
    parts      = strsplit(strtrim(line));
    trial_type = parts{5};
    re_rgb     = '([0-9.+\-]+)_G([0-9.+\-]+)_B([0-9.+\-]+)';
    ref_rgb    = parseRGB(line, ['Ref_R'   re_rgb]);
    comp_rgb   = parseRGB(line, ['Comp_R'  re_rgb]);   % won't match Comp2_R
    comp2_rgb  = parseRGB(line, ['Comp2_R' re_rgb]);
end

function rgb = parseRGB(line, pat)
    tok = regexp(line, pat, 'tokens', 'once');
    if isempty(tok)
        rgb = [];
    else
        rgb = [str2double(tok{1}), str2double(tok{2}), str2double(tok{3})];
    end
end

function validateSessionFile(file_path, subject_init, session_today)
    fid = fopen(file_path, 'r');
    if fid == -1
        error('Cannot open session file: %s', file_path);
    end
    found_init    = '';
    found_session = NaN;
    for i = 1:4
        line = fgetl(fid);
        if ~ischar(line), break; end
        if startsWith(line, 'Subject initial:')
            found_init = strtrim(extractAfter(line, ':'));
        elseif startsWith(line, 'Session:')
            found_session = str2double(strtrim(extractAfter(line, ':')));
        end
    end
    fclose(fid);
    if ~strcmp(found_init, subject_init)
        error('Subject initials mismatch: file has "%s", entered "%s".', ...
            found_init, subject_init);
    end
    if found_session ~= session_today
        error('Session number mismatch: file has %d, entered %d.', ...
            found_session, session_today);
    end
end
