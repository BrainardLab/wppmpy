classdef WPPMCommunicator < handle
% WPPMCommunicator  Text-file-based trial communication for WPPM experiments.
%
% Sender and recipient coordinate by appending lines to a shared session
% file on a network disk.  This mirrors the Python CommunicateViaTextFile
% class in ellipsoids-elife2025/analysis/utils_communication.py.
%
% Recipient-side usage (production — separate receive and send):
%
%   fullPath = WPPMCommunicator.waitForSessionFile(pathSub, pattern);
%   comm = WPPMCommunicator(fullPath);
%   comm.confirmCommunication();
%   while ~comm.terminate
%       [trialType, refRGB, compRGB, comp2RGB] = comm.receiveStimuli();
%       if comm.terminate, break; end
%       response = <run experiment and get response>;
%       comm.sendResponse(trialType, refRGB, compRGB, response, comp2RGB);
%   end
%
% Recipient-side usage (test — random responses via confirmRGBvals):
%
%   fullPath = WPPMCommunicator.waitForSessionFile(pathSub, pattern);
%   comm = WPPMCommunicator(fullPath);
%   comm.confirmCommunication();
%   while ~comm.terminate
%       comm.confirmRGBvals(0.1);
%   end

    properties
        filePath   (1,:) char             % full path to shared session file
        retryDelay (1,1) double = 1/60    % polling interval in seconds
        timeout    (1,1) double = 1200    % max wait in seconds (20 min)
        terminate  (1,1) logical = false  % true when 'Done' received
    end

    % ------------------------------------------------------------------ %
    methods

        function obj = WPPMCommunicator(filePath, options)
        % WPPMCommunicator  Construct communicator for an existing session file.
        %
        %   comm = WPPMCommunicator(fullFilePath)
        %   comm = WPPMCommunicator(fullFilePath, retryDelay=1/60, timeout=1200)
            arguments
                filePath            (1,:) char
                options.retryDelay  (1,1) double = 1/60
                options.timeout     (1,1) double = 1200
            end
            obj.filePath   = filePath;
            obj.retryDelay = options.retryDelay;
            obj.timeout    = options.timeout;
        end

        % -------------------------------------------------------------- %
        function confirmCommunication(obj)
        % confirmCommunication  Wait for Set_Up_to_Communicate, reply Ready_To_Communicate.
            t = tic;
            while true
                if strcmp(obj.lastWord(), 'Set_Up_to_Communicate')
                    obj.appendMessage('Ready_To_Communicate');
                    return;
                end
                if toc(t) > obj.timeout
                    error('WPPMCommunicator:timeout', ...
                        'Timed out waiting for Set_Up_to_Communicate.');
                end
                pause(obj.retryDelay);
            end
        end

        % -------------------------------------------------------------- %
        function [trialType, refRGB, compRGB, comp2RGB] = receiveStimuli(obj)
        % receiveStimuli  Wait for the next trial command from the conductor.
        %
        %   Returns trialType (char), refRGB (1x3), compRGB (1x3), comp2RGB (1x3 or []).
        %   Sets obj.terminate = true and returns empty arrays if 'Done' received.
        %   Handles 'Break' transparently (pauses, sends 'Resume', keeps polling).
            trialType = '';  refRGB = [];  compRGB = [];  comp2RGB = [];
            t = tic;
            while true
                lastLine = obj.lastLine();
                lastWord = WPPMCommunicator.lastWordOf(lastLine);

                if strcmp(lastWord, 'Done')
                    obj.terminate = true;
                    return;

                elseif strcmp(lastWord, 'Break')
                    pause(1);
                    obj.appendMessage('Resume');
                    t = tic;

                elseif strcmp(lastWord, 'Image_Display')
                    [trialType, refRGB, compRGB, comp2RGB] = ...
                        WPPMCommunicator.parseTrialLine(lastLine);
                    return;

                else
                    if toc(t) > obj.timeout
                        error('WPPMCommunicator:timeout', ...
                            'Timed out waiting for Image_Display.');
                    end
                    pause(obj.retryDelay);
                end
            end
        end

        % -------------------------------------------------------------- %
        function sendResponse(obj, trialType, refRGB, compRGB, response, comp2RGB)
        % sendResponse  Send Image_Confirmed with trial info and response to the conductor.
        %
        %   response: integer (0 or 1)
        %   comp2RGB: (1x3) for suprathreshold trials, or [] for oddity trials
            if nargin < 6 || isempty(comp2RGB)
                comp2Str = '';
            else
                comp2Str = sprintf('Comp2_R%.8f_G%.8f_B%.8f ', ...
                    comp2RGB(1), comp2RGB(2), comp2RGB(3));
            end
            msg = sprintf( ...
                '%s Ref_R%.8f_G%.8f_B%.8f Comp_R%.8f_G%.8f_B%.8f %sResp%d Image_Confirmed', ...
                trialType, ...
                refRGB(1),  refRGB(2),  refRGB(3), ...
                compRGB(1), compRGB(2), compRGB(3), ...
                comp2Str, response);
            obj.appendMessage(msg);
        end

        % -------------------------------------------------------------- %
        function [trialType, refRGB, compRGB] = confirmRGBvals(obj, responseDelay)
        % confirmRGBvals  Test method: receive stimuli, send random response.
        %
        %   Calls receiveStimuli and sendResponse with a random response in between.
        %   Use receiveStimuli and sendResponse directly for production code.
        %   Returns trialType (char), refRGB (1x3), compRGB (1x3); empty on termination.
            if nargin < 2, responseDelay = 0.1; end
            [trialType, refRGB, compRGB, comp2RGB] = obj.receiveStimuli();
            if obj.terminate, return; end
            pause(responseDelay);
            obj.sendResponse(trialType, refRGB, compRGB, randi([0 1]), comp2RGB);
        end

        % -------------------------------------------------------------- %
        function appendMessage(obj, message)
        % appendMessage  Append a timestamped, hostname-tagged line to the session file.
            [~, hostname] = system('hostname');
            hostname  = strtrim(hostname);
            timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
            fid = fopen(obj.filePath, 'a');
            if fid == -1
                error('WPPMCommunicator:fileError', ...
                    'Cannot open file for appending: %s', obj.filePath);
            end
            fprintf(fid, '%s - %s: %s\n', timestamp, hostname, message);
            fclose(fid);
        end

    end % public methods

    % ------------------------------------------------------------------ %
    methods (Access = private)

        function line = lastLine(obj)
        % Return the last non-empty line of the session file, or '' if unreadable.
            fid = fopen(obj.filePath, 'r');
            if fid == -1
                line = '';   % file locked mid-sync; caller will retry
                return;
            end
            line = '';
            while ~feof(fid)
                l = fgetl(fid);
                if ischar(l) && ~isempty(strtrim(l))
                    line = strtrim(l);
                end
            end
            fclose(fid);
        end

        function word = lastWord(obj)
            word = WPPMCommunicator.lastWordOf(obj.lastLine());
        end

    end % private methods

    % ------------------------------------------------------------------ %
    methods (Static)

        function filePath = waitForSessionFile(pathSub, pattern, timeout)
        % waitForSessionFile  Poll until a matching session file appears.
        %
        %   filePath = WPPMCommunicator.waitForSessionFile(pathSub, pattern)
        %   filePath = WPPMCommunicator.waitForSessionFile(pathSub, pattern, timeout)
        %
        %   Blocks until a file matching the glob pattern appears in pathSub,
        %   then returns its full path.  timeout defaults to 120 s.
            if nargin < 3, timeout = 120; end
            fprintf('Waiting for session file in %s matching %s ...\n', pathSub, pattern);
            t = tic;
            while true
                files = dir(fullfile(pathSub, pattern));
                if ~isempty(files)
                    break;
                end
                if toc(t) > timeout
                    error('WPPMCommunicator:timeout', ...
                        'Timed out waiting for session file in %s matching %s', ...
                        pathSub, pattern);
                end
                pause(0.5);
            end
            [~, idx] = max([files.datenum]);
            filePath = fullfile(pathSub, files(idx).name);
            fprintf('Found session file: %s\n', files(idx).name);
        end

        function word = lastWordOf(line)
        % lastWordOf  Return the last whitespace-delimited token of a string.
            parts = strsplit(strtrim(line));
            if isempty(parts) || isempty(parts{1})
                word = '';
            else
                word = parts{end};
            end
        end

        function [trialType, refRGB, compRGB, comp2RGB] = parseTrialLine(line)
        % parseTrialLine  Extract trial type and RGB triplets from an Image_Display line.
            parts     = strsplit(strtrim(line));
            trialType = parts{5};
            re        = '([0-9.+\-eE]+)_G([0-9.+\-eE]+)_B([0-9.+\-eE]+)';
            refRGB   = WPPMCommunicator.parseRGB(line, ['Ref_R'   re]);
            compRGB  = WPPMCommunicator.parseRGB(line, ['Comp_R'  re]);
            comp2RGB = WPPMCommunicator.parseRGB(line, ['Comp2_R' re]);
        end

    end % public static methods

    % ------------------------------------------------------------------ %
    methods (Static, Access = private)

        function rgb = parseRGB(line, pat)
            tok = regexp(line, pat, 'tokens', 'once');
            if isempty(tok)
                rgb = [];
            else
                rgb = [str2double(tok{1}), str2double(tok{2}), str2double(tok{3})];
            end
        end

    end % private static methods

end % classdef
