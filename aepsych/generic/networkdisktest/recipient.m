% recipient.m
% MATLAB test stand-in for recipient.py.
%
% Reads network_disk_path from local_config.json in the same directory.
% Copy local_config.json.template -> local_config.json and fill in the path.
%
% Responses are random (0 or 1).

%% Add communication class to path
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..', 'matlab'));

%% Read config
cfg = jsondecode(fileread(fullfile(here, 'local_config.json')));
network_disk_path = cfg.network_disk_path;

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

%% Wait for sender to create the session file
path_sub = fullfile(network_disk_path, sprintf('sub%d', subject_id));
pat      = sprintf('sub%d_%s*session%d*.txt', subject_id, subject_init, session_today);

full_file_path = WPPMCommunicator.waitForSessionFile(path_sub, pat);

%% Run communication
comm = WPPMCommunicator(full_file_path);

fprintf('Waiting for initialization command...\n');
comm.confirmCommunication();
fprintf('Initialization confirmed.\n');

trial_counter = 0;
while ~comm.terminate
    comm.confirmRGBvals(0.1);
    trial_counter = trial_counter + 1;
    fprintf('Trial #%d confirmed.\n', trial_counter);
end

fprintf('Communication complete (%d trials).\n', trial_counter);
