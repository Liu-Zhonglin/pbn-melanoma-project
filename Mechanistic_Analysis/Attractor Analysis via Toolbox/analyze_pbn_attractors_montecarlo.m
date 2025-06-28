% =========================================================================
% SCRIPT: run_attractor_analysis_v6.m
%
% PURPOSE:
% Performs a comprehensive attractor analysis using a robust, toolbox-driven
% Monte Carlo method.
%
% v6 REVISIONS:
% - Updated to call 'calculate_pbn_attractor_landscape_v3.m' to use the
%   version with the corrected nested function placement.
%
% Last updated: June 23, 2025
% =========================================================================

clc; clear; close all;

%% --- 1. SETUP AND CONFIGURATION ---
fprintf('--- SETUP AND CONFIGURATION ---\n');

% Add PBN toolbox to path
toolbox_path = '/Users/liuzhonglin/Desktop/PBN_Melanoma_Project/Improved/Toolbox/pbn-matlab-toolbox';
addpath(genpath(toolbox_path));

% Setup directories
modelDir = pwd;
outputDir = fullfile(modelDir, 'Attractor_Analysis_Results_v6');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

% Configuration for Monte Carlo simulation
num_samples = 1000000 ; % Use a high number for robust results

fprintf('Setup complete. Results will be saved to: %s\n\n', outputDir);

%% --- 2. SETUP PARALLEL ENVIRONMENT ---
fprintf('--- PREPARING PARALLEL ENVIRONMENT ---\n');
if isempty(gcp('nocreate')), parpool; end
poolobj = gcp;

% Get the list of all toolbox .m files
toolbox_files = dir(fullfile(toolbox_path, '**', '*.m'));
toolbox_file_list = fullfile({toolbox_files.folder}, {toolbox_files.name});

% Get the full path to our custom helper function (the v3 version)
helper_function_path = which('calculate_pbn_attractor_landscape_v3.m');
if isempty(helper_function_path)
    error('CRITICAL: The helper function "calculate_pbn_attractor_landscape_v3.m" was not found on the MATLAB path.');
end

% Combine the toolbox files AND our custom helper function into one list
all_files_to_attach = [toolbox_file_list, {helper_function_path}];

% Attach the complete list of files to the workers
addAttachedFiles(poolobj, all_files_to_attach);
fprintf('Attached %d toolbox files and 1 custom helper function to the parallel workers.\n\n', length(toolbox_file_list));


%% --- 3. ATTRACTOR ANALYSIS ---
fprintf('--- STARTING: Attractor Landscape Analysis ---\n');

% --- Analyze the Responder Model ---
fprintf('--> Analyzing RESPONDER PBN Model...\n');
responder_attractor_table = calculate_pbn_attractor_landscape_v3('responder_PBN_model_mi_loose.json', num_samples);
if ~isempty(responder_attractor_table)
    writetable(responder_attractor_table, fullfile(outputDir, 'attractors_responder.csv'));
    fprintf('--> COMPLETE: Found %d unique attractors in the responder model.\n\n', height(responder_attractor_table));
else
    fprintf('--> COMPLETE: No attractors found for the responder model.\n\n');
end

% --- Analyze the Non-Responder Model ---
fprintf('--> Analyzing NON-RESPONDER PBN Model...\n');
non_responder_attractor_table = calculate_pbn_attractor_landscape_v3('non_responder_PBN_model_mi_loose.json', num_samples);
if ~isempty(non_responder_attractor_table)
    writetable(non_responder_attractor_table, fullfile(outputDir, 'attractors_non_responder.csv'));
    fprintf('--> COMPLETE: Found %d unique attractors in the non-responder model.\n\n', height(non_responder_attractor_table));
else
    fprintf('--> COMPLETE: No attractors found for the non-responder model.\n\n');
end

fprintf('====== ATRACTOR ANALYSIS SCRIPT IS COMPLETE ======\n');