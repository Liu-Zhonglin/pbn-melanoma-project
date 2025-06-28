% =========================================================================
% SCRIPT FOR PREPARING THE GSE78220 MELANOMA RNA-SEQ DATASET (v4)
% =========================================================================
%
% Description:
% This script loads the public RNA-seq count matrix from GEO (GSE78220)
% and its corresponding annotation file. It performs the following steps:
% 1. Converts GeneIDs to Gene Symbols.
% 2. Classifies each sample as "Responder" or "Non-Responder" based on
%    pre-defined clinical outcomes.
% 3. Excludes one sample identified as "on-treatment" to ensure the
%    dataset only contains pre-treatment profiles.
% 4. Renames the columns (samples) to include their classification label
%    (e.g., 'GSMXXXX_responder').
% 5. Saves the final, clean count matrix, ready for binarization.
%
% v4 Update: Major revision to focus solely on preparing the GSE78220
%            dataset for the anti-PD-1 resistance study. Removed all
%            code related to personal data integration.
%
% Author: Your Name / AI Assistant
% Date: June 2025
%
clear; clc; close all;

%% ========================================================================
%  SECTION 0: CONFIGURATION - SET YOUR FILE NAMES HERE
% =========================================================================
publicDataFile = 'GSE78220_raw_counts_GRCh38.p13_NCBI.tsv';
annotationFile = 'Human.GRCh38.p13.annot.tsv';

% This will be the input for the next script in the pipeline (Binarization.R)
outputCountFile   = 'final_clean_counts.csv';
% This file is for your records, to confirm the classification
outputClassFile   = 'sample_classification.csv';


%% ========================================================================
%  SECTION 1: LOAD PUBLIC DATA AND ANNOTATION
% =========================================================================
fprintf('SECTION 1: Loading public data (GSE78220) and annotation file...\n');

try
    opts_public = detectImportOptions(publicDataFile, 'FileType', 'text');
    opts_public.Delimiter = '\t';
    publicData = readtable(publicDataFile, opts_public);
    publicData.Properties.VariableNames{1} = 'GeneID';
catch ME
    error('Failed to load public data file: %s. Please ensure it is in the working directory.\nError: %s', publicDataFile, ME.message);
end

try
    opts_annot = detectImportOptions(annotationFile, 'FileType', 'text');
    opts_annot.Delimiter = '\t';
    annotationData = readtable(annotationFile, opts_annot);
catch ME
    error('Failed to load annotation file: %s. Please ensure it is in the working directory.\nError: %s', annotationFile, ME.message);
end

fprintf('Data loading complete.\n\n');


%% ========================================================================
%  SECTION 2: PREPARE ANNOTATION AND CONVERT GENEIDS
% =========================================================================
fprintf('SECTION 2: Preparing annotation and converting GeneIDs to Symbols...\n');

% Clean annotation map
annotMap = annotationData(:, {'GeneID', 'Symbol'});
annotMap(ismissing(annotMap.Symbol), :) = [];
[~, uniqueIdx] = unique(annotMap.GeneID, 'stable');
annotMap = annotMap(uniqueIdx, :);

% Join with public data to add Gene Symbols
publicData_annotated = innerjoin(publicData, annotMap, 'Keys', 'GeneID');

% Make gene symbols unique and set as row names
uniqueSymbols = makeUniqueNames(publicData_annotated.Symbol);
publicData_annotated.Properties.RowNames = uniqueSymbols;
publicData_annotated.GeneID = [];
publicData_annotated.Symbol = [];

fprintf('Gene ID conversion successful.\n\n');


%% ========================================================================
%  SECTION 3: CLASSIFY SAMPLES AND RENAME COLUMNS
% =========================================================================
fprintf('SECTION 3: Classifying samples based on clinical response...\n');

% --- Define sample groups based on clinical metadata analysis ---
responder_gsm = { ...
    'GSM2069824', 'GSM2069825', 'GSM2069826', 'GSM2069827', 'GSM2069829', ...
    'GSM2069830', 'GSM2069833', 'GSM2069835', 'GSM2069837', 'GSM2069842', ...
    'GSM2069843', 'GSM2069844', 'GSM2069848', 'GSM2069849', 'GSM2069850'};

non_responder_gsm = { ...
    'GSM2069823', 'GSM2069828', 'GSM2069831', 'GSM2069832', 'GSM2069834', ...
    'GSM2069838', 'GSM2069839', 'GSM2069840', 'GSM2069841', 'GSM2069845', ...
    'GSM2069846', 'GSM2069847'};

excluded_gsm = {'GSM2069836'}; % This sample is on-treatment

% --- Loop through columns to rename and create classification list ---
original_colnames = publicData_annotated.Properties.VariableNames;
new_colnames = original_colnames;
classification_list = cell(length(original_colnames), 2); % {SampleID, Classification}

for i = 1:length(original_colnames)
    gsm_id = original_colnames{i};
    classification_list{i, 1} = gsm_id;

    if ismember(gsm_id, responder_gsm)
        new_colnames{i} = sprintf('%s_responder', gsm_id);
        classification_list{i, 2} = 'Responder';
    elseif ismember(gsm_id, non_responder_gsm)
        new_colnames{i} = sprintf('%s_non_responder', gsm_id);
        classification_list{i, 2} = 'Non-Responder';
    elseif ismember(gsm_id, excluded_gsm)
        new_colnames{i} = sprintf('%s_excluded', gsm_id);
        classification_list{i, 2} = 'Excluded';
        fprintf('INFO: Sample %s marked for exclusion (on-treatment sample).\n', gsm_id);
    else
        new_colnames{i} = sprintf('%s_unclassified', gsm_id);
        classification_list{i, 2} = 'Unclassified';
        fprintf('WARNING: Sample %s was not found in any group and will be excluded.\n', gsm_id);
    end
end

% Apply the new, descriptive column names
publicData_annotated.Properties.VariableNames = new_colnames;

% Create a table for the classification metadata
classification_table = cell2table(classification_list, ...
    'VariableNames', {'SampleID', 'Classification'});

fprintf('Sample classification complete.\n\n');


%% ========================================================================
%  SECTION 4: FINALIZE DATA AND SAVE RESULTS
% =========================================================================
fprintf('SECTION 4: Removing excluded samples and saving final files...\n');

% Create the final data table by removing any columns marked for exclusion
final_data = publicData_annotated;
columns_to_remove = contains(final_data.Properties.VariableNames, '_excluded') | ...
                    contains(final_data.Properties.VariableNames, '_unclassified');
final_data(:, columns_to_remove) = [];

% Save the final count matrix
writetable(final_data, outputCountFile, 'WriteRowNames', true);

% Save the classification file for your records
writetable(classification_table, outputClassFile);

fprintf('------------------------------------------------------------\n');
fprintf('PROCESS COMPLETE\n');
fprintf('Final clean count matrix saved to: %s\n', outputCountFile);
fprintf(' -> This file contains %d genes and %d samples.\n', height(final_data), width(final_data));
fprintf('Sample classification map saved to: %s\n', outputClassFile);
fprintf('------------------------------------------------------------\n');


%% ========================================================================
%  HELPER FUNCTION
% =========================================================================
function uniqueNames = makeUniqueNames(names)
    % This function handles duplicate gene symbols by appending _1, _2, etc.
    uniqueNames = cell(size(names));
    counts = containers.Map('KeyType','char','ValueType','int32');
    for i = 1:length(names)
        name = names{i};
        if isKey(counts, name)
            counts(name) = counts(name) + 1;
            uniqueNames{i} = sprintf('%s_%d', name, counts(name)-1);
        else
            counts(name) = 1;
            uniqueNames{i} = name;
        end
    end
end