% =========================================================================
% SCRIPT: analyze_and_decode_attractors_v2.m
%
% PURPOSE:
%   Reads the output from the PBN attractor analysis, decodes the
%   attractor states into binary gene expression patterns, classifies them
%   into predefined biological phenotypes, and generates a comprehensive
%   summary report comparing the responder and non-responder models.
%
% v2 REVISIONS:
%   - Added clc; clear; close all; for a clean workspace.
%   - CRITICAL FIX: Corrected the hardcoded GENE_LIST to match the PBN models.
%   - Refined the phenotype classification logic to match the user's paper.
%   - Made CSV reading more robust for different MATLAB versions.
% =========================================================================

clc;
clear;
close all;

fprintf('--- Starting Attractor Landscape Analysis and Decoding (v2) ---\n\n');

%% --- 1. CONFIGURATION ---

% --- Define the ordered list of genes used in the PBN model ---
% NOTE: This order has been corrected to exactly match the JSON model files.
GENE_LIST = {
    'AXL', 'JUN', 'LOXL2', 'MAP2K3', 'MAPK3', 'NFATC1', 'NRAS', ...
    'PIK3CB', 'RELA', 'ROR2', 'TAGLN', 'WNT5A'
};
NUM_GENES = length(GENE_LIST);
fprintf('Gene list loaded successfully (%d genes).\n', NUM_GENES);
disp(strjoin(GENE_LIST, ', '));
fprintf('\n');

% --- Define key genes for phenotype classification based on the IPRES signature ---
KEY_GENES.AXL = find(strcmp(GENE_LIST, 'AXL'));
KEY_GENES.WNT5A = find(strcmp(GENE_LIST, 'WNT5A'));
KEY_GENES.ROR2 = find(strcmp(GENE_LIST, 'ROR2'));

% --- Input / Output Files ---
outputDir = fullfile(pwd, 'Phenotype_Analysis_Results');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

responder_csv_in = 'attractors_responder.csv';
non_responder_csv_in = 'attractors_non_responder.csv';

responder_csv_out = fullfile(outputDir, 'attractors_responder_decoded.csv');
non_responder_csv_out = fullfile(outputDir, 'attractors_non_responder_decoded.csv');
report_file_out = fullfile(outputDir, 'comparison_report_final.txt');

%% --- 2. PROCESS AND DECODE ATTRACTOR FILES ---

fprintf('--> Processing Responder Model...\n');
if ~exist(responder_csv_in, 'file')
    error('File not found: %s. Please ensure it is in the same directory as the script.', responder_csv_in);
end
responder_table = process_attractor_file(responder_csv_in, NUM_GENES, KEY_GENES);
writetable(responder_table, responder_csv_out);
fprintf('    Decoded responder results saved to: %s\n\n', responder_csv_out);


fprintf('--> Processing Non-Responder Model...\n');
if ~exist(non_responder_csv_in, 'file')
    error('File not found: %s. Please ensure it is in the same directory as the script.', non_responder_csv_in);
end
non_responder_table = process_attractor_file(non_responder_csv_in, NUM_GENES, KEY_GENES);
writetable(non_responder_table, non_responder_csv_out);
fprintf('    Decoded non-responder results saved to: %s\n\n', non_responder_csv_out);


%% --- 3. GENERATE COMPARISON REPORT ---
fprintf('--> Generating final comparison report...\n');
generate_comparison_report(report_file_out, responder_table, non_responder_table, GENE_LIST);
fprintf('    Report saved to: %s\n\n', report_file_out);

fprintf('====== ANALYSIS COMPLETE ======\n');


%% --- HELPER FUNCTIONS ---

function processed_table = process_attractor_file(filename, num_genes, key_genes)
    % Reads a CSV file of attractors and adds decoded binary states and phenotypes.
    opts = detectImportOptions(filename);
    opts = setvartype(opts, 'AttractorStates_Decimal', 'string');
    tbl = readtable(filename, opts);

    if isempty(tbl)
        processed_table = table(); % Return an empty table if no attractors were found
        return;
    end

    % Make robust check for cell vs. string array for different MATLAB versions
    is_cell_column = iscell(tbl.AttractorStates_Decimal);

    num_rows = height(tbl);
    phenotypes = cell(num_rows, 1);
    key_gene_states_str = cell(num_rows, 1);
    binary_states_str = cell(num_rows, 1);

    for i = 1:num_rows
        if is_cell_column
            decimal_states_str = tbl.AttractorStates_Decimal{i};
        else % It's a string array
            decimal_states_str = tbl.AttractorStates_Decimal(i);
        end
        
        decimal_states = str2double(strsplit(char(decimal_states_str), ','));
        
        first_state_decimal = decimal_states(1);
        binary_vector = decimal_to_binary_vector(first_state_decimal, num_genes);

        phenotypes{i} = classify_phenotype(binary_vector, key_genes);
        key_gene_states_str{i} = format_key_gene_states(binary_vector, key_genes);
        binary_states_str{i} = binary_vector_to_string(binary_vector);
    end
    
    processed_table = addvars(tbl, phenotypes, key_gene_states_str, binary_states_str, ...
        'NewVariableNames', {'Phenotype', 'KeyGeneStates', 'BinaryState'});
end

function binary_vector = decimal_to_binary_vector(dec_val, num_genes)
    % Converts a decimal state to its binary vector representation.
    bin_str = dec2bin(dec_val, num_genes);
    binary_vector = bin_str - '0';
end

function phenotype = classify_phenotype(binary_vector, key_genes)
    % Classifies a binary state vector based on the user's paper.
    is_axl_on = (binary_vector(key_genes.AXL) == 1);
    is_wnt5a_on = (binary_vector(key_genes.WNT5A) == 1);
    is_ror2_on = (binary_vector(key_genes.ROR2) == 1);

    % Order of checks is important to handle specific definitions first.
    if is_wnt5a_on && is_ror2_on
        phenotype = 'Resistant (WNT-Driven)';
    elseif is_axl_on && ~is_wnt5a_on && is_ror2_on
        phenotype = 'Resistant (AXL-Driven)';
    elseif ~is_axl_on && ~is_wnt5a_on && is_ror2_on
        phenotype = 'Sensitive';
    else
        phenotype = 'Unclassified';
    end
end

function state_str = format_key_gene_states(binary_vector, key_genes)
    % Creates a formatted string like (AXL=1, WNT5A=0, ROR2=1)
    axl_state = binary_vector(key_genes.AXL);
    wnt5a_state = binary_vector(key_genes.WNT5A);
    ror2_state = binary_vector(key_genes.ROR2);
    state_str = sprintf('(AXL=%d, WNT5A=%d, ROR2=%d)', axl_state, wnt5a_state, ror2_state);
end

function bin_str = binary_vector_to_string(binary_vector)
    % Converts a numeric binary vector [1 0 1] to a string '101'
    bin_str = sprintf('%d', binary_vector);
end

function generate_comparison_report(filename, res_tbl, non_res_tbl, gene_list)
    % Generates a detailed text report comparing the two models.
    fid = fopen(filename, 'w');
    if fid == -1, error('Could not open file for writing: %s', filename); end
    
    fprintf(fid, '==========================================================\n');
    fprintf(fid, ' PBN Attractor Landscape Analysis: Responder vs. Non-Responder\n');
    fprintf(fid, '==========================================================\n\n');
    fprintf(fid, 'Date Generated: %s\n\n', datestr(now));

    % Section 1: Global Landscape Comparison
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '1. Global Landscape Properties\n');
    fprintf(fid, '----------------------------------------\n');
    
    res_total = height(res_tbl);
    non_res_total = height(non_res_tbl);
    res_dom_prob = ifthen(res_total > 0, res_tbl.Estimated_Probability(1) * 100, 0);
    non_res_dom_prob = ifthen(non_res_total > 0, non_res_tbl.Estimated_Probability(1) * 100, 0);
    
    fprintf(fid, '%-40s | %-25s | %-25s\n', 'Metric', 'Responder (Sensitive)', 'Non-Responder (Resistant)');
    fprintf(fid, '%s\n', repmat('-', 1, 98));
    fprintf(fid, '%-40s | %-25d | %-25d\n', 'Total Unique Attractors Found', res_total, non_res_total);
    fprintf(fid, '%-40s | %-25.2f%% | %-25.2f%%\n', 'Probability of Most Dominant Attractor', res_dom_prob, non_res_dom_prob);
    fprintf(fid, '\n\n');
    
    % Section 2: Definitions
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '2. Phenotype Definitions\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, 'Gene Order: %s\n\n', strjoin(gene_list, ', '));
    fprintf(fid, 'Phenotypes are classified based on the state of key IPRES genes:\n');
    fprintf(fid, ' - Sensitive:              AXL=0, WNT5A=0, ROR2=1\n');
    fprintf(fid, ' - Resistant (WNT-Driven): WNT5A=1, ROR2=1\n');
    fprintf(fid, ' - Resistant (AXL-Driven): AXL=1, WNT5A=0, ROR2=1\n');
    fprintf(fid, '\n\n');

    % Section 3: Responder Details
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '3. Responder Model: Top Attractors\n');
    fprintf(fid, '----------------------------------------\n');
    print_attractor_table_to_file(fid, res_tbl, 10);
    fprintf(fid, '\n\n');
    
    % Section 4: Non-Responder Details
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '4. Non-Responder Model: Top Attractors\n');
    fprintf(fid, '----------------------------------------\n');
    print_attractor_table_to_file(fid, non_res_tbl, 10);
    
    fclose(fid);
end

function print_attractor_table_to_file(fid, tbl, num_to_print)
    % Helper to print a formatted table of attractors to the report file.
    if isempty(tbl)
        fprintf(fid, 'No attractors found for this model.\n');
        return;
    end
    
    max_rows = min(num_to_print, height(tbl));
    
    fprintf(fid, '%-10s | %-8s | %-25s | %-30s | %s\n', 'Prob. (%)', 'Size', 'Phenotype', 'Key Gene States', 'Decimal State(s)');
    fprintf(fid, '%s\n', repmat('-', 1, 105));
    
    for i = 1:max_rows
        prob_pct = tbl.Estimated_Probability(i) * 100;
        att_size = tbl.Attractor_Size(i);
        phenotype = tbl.Phenotype{i};
        key_states = tbl.KeyGeneStates{i};
        dec_states = tbl.AttractorStates_Decimal(i); % Use () for string array
        
        fprintf(fid, '%-10.2f | %-8d | %-25s | %-30s | %s\n', prob_pct, att_size, phenotype, key_states, char(dec_states));
    end
end

function out = ifthen(cond, true_val, false_val)
    % A simple inline-if function to avoid clutter
    if cond
        out = true_val;
    else
        out = false_val;
    end
end
