% =========================================================================
% SCRIPT: analyze_pbn_rewiring_and_influence.m
%
% PURPOSE:
%   Performs a detailed comparative analysis of two PBN models. It loads
%   the JSON models, calculates the global influence matrix for each using
%   the formal definition of influence (functional sensitivity), and
%   extracts the specific Boolean functions and their probabilities to
%   reveal topological and logical rewiring between the two networks.
%
% v2 REVISIONS:
%   - The 'calculate_global_influence' function has been completely
%     rewritten to use the formal, functional sensitivity definition of
%     influence, as requested by the user.
% =========================================================================

clc;
clear;
close all;

fprintf('--- Starting PBN Rewiring and Influence Analysis (v2) ---\n\n');

%% --- 1. CONFIGURATION ---
RESPONDER_MODEL_FILE = 'responder_PBN_model_mi_loose.json';
NON_RESPONDER_MODEL_FILE = 'non_responder_PBN_model_mi_loose.json';
OUTPUT_REPORT_FILE = 'rewiring_and_influence_report.txt';

% --- Load Models ---
fprintf('--> Loading PBN models...\n');
try
    responder_model = jsondecode(fileread(RESPONDER_MODEL_FILE));
    non_responder_model = jsondecode(fileread(NON_RESPONDER_MODEL_FILE));
catch ME
    error('Failed to load or parse JSON model files. Error: %s', ME.message);
end

% Extract gene list (assuming it's the same for both models)
gene_list = fieldnames(responder_model.nodes);
num_genes = length(gene_list);
fprintf('    Models loaded successfully for %d genes.\n\n', num_genes);


%% --- 2. CALCULATE GLOBAL INFLUENCE ---
fprintf('--> Calculating Global Influence Matrices (using Functional Sensitivity)...\n');
fprintf('    This may take a moment...\n');
responder_influence_matrix = calculate_global_influence(responder_model, gene_list);
non_responder_influence_matrix = calculate_global_influence(non_responder_model, gene_list);
fprintf('    Influence calculation complete.\n\n');


%% --- 3. GENERATE REPORT ---
fprintf('--> Generating final analysis report...\n');
generate_rewiring_report( ...
    OUTPUT_REPORT_FILE, ...
    gene_list, ...
    responder_model, ...
    non_responder_model, ...
    responder_influence_matrix, ...
    non_responder_influence_matrix ...
);
fprintf('    Report saved to: %s\n\n', OUTPUT_REPORT_FILE);

fprintf('====== ANALYSIS COMPLETE ======\n');


%% --- HELPER FUNCTIONS ---

function influence_matrix = calculate_global_influence(model, gene_list)
    % Calculates the formal functional influence of each gene on every other gene.
    % Influence I(k->i) is the probability that toggling regulator k changes the
    % output for target i, averaged across all possible predictor functions for i.
    num_genes = length(gene_list);
    influence_matrix = zeros(num_genes, num_genes); % Rows: Regulators, Cols: Targets

    for i = 1:num_genes % For each TARGET gene (column)
        target_gene = gene_list{i};
        node_data = model.nodes.(target_gene);
        
        % This map will store the total influence of all regulators on this specific target
        total_influence_on_target = containers.Map('KeyType', 'char', 'ValueType', 'double');

        % Iterate through each possible predictor function for the target gene
        for j = 1:length(node_data.functions)
            func_info = node_data.functions(j);
            func_prob = func_info.probability;
            func_str = func_info.function;
            inputs = func_info.inputs;
            num_inputs = length(inputs);

            if num_inputs == 0, continue; end

            % For each input to this function, calculate its sensitivity
            for k = 1:num_inputs 
                regulator_gene = inputs{k};
                
                % Calculate the sensitivity of this function to this one regulator
                flip_count = 0;
                num_input_states = 2^num_inputs;
                
                for state_idx = 0:(num_input_states - 1)
                    % Generate the binary input vector for the function's inputs
                    input_vector = decimal_to_binary_vector(state_idx, num_inputs);
                    
                    % Evaluate the function with the original input
                    output1 = evaluate_boolean_function(func_str, inputs, input_vector);
                    
                    % Toggle the value of the regulator we are testing (at index k)
                    toggled_input_vector = input_vector;
                    toggled_input_vector(k) = ~toggled_input_vector(k);
                    
                    % Evaluate the function with the toggled input
                    output2 = evaluate_boolean_function(func_str, inputs, toggled_input_vector);
                    
                    if output1 ~= output2
                        flip_count = flip_count + 1;
                    end
                end
                
                % Sensitivity is the fraction of times the output flipped
                sensitivity = flip_count / num_input_states;
                
                % Add the probability-weighted influence to the total for that regulator
                if isKey(total_influence_on_target, regulator_gene)
                    total_influence_on_target(regulator_gene) = total_influence_on_target(regulator_gene) + (sensitivity * func_prob);
                else
                    total_influence_on_target(regulator_gene) = (sensitivity * func_prob);
                end
            end
        end
        
        % Populate the main influence matrix for this target gene
        all_regulators = keys(total_influence_on_target);
        for k = 1:length(all_regulators)
            regulator_gene = all_regulators{k};
            regulator_idx = find(strcmp(gene_list, regulator_gene), 1);
            if ~isempty(regulator_idx)
                influence_matrix(regulator_idx, i) = total_influence_on_target(regulator_gene);
            end
        end
    end
end

function output = evaluate_boolean_function(func_str, inputs, input_vector)
    % Evaluates a boolean function string given input names and their values.
    eval_str = func_str;
    for i = 1:length(inputs)
        % Use regexprep with word boundaries \< and \> to avoid
        % replacing substrings (e.g., preventing 'A' from matching 'AXL').
        pattern = ['\<' inputs{i} '\>'];
        replacement = num2str(input_vector(i));
        eval_str = regexprep(eval_str, pattern, replacement);
    end
    % Convert to MATLAB logical syntax if not already
    eval_str = strrep(eval_str, 'AND', '&');
    eval_str = strrep(eval_str, 'OR', '|');
    eval_str = strrep(eval_str, 'NOT', '~');
    
    % Handle edge case of a constant '0' or '1' function
    if strcmp(eval_str, '0')
        output = 0;
    elseif strcmp(eval_str, '1')
        output = 1;
    else
        output = eval(eval_str);
    end
end

function binary_vector = decimal_to_binary_vector(dec_val, num_bits)
    % Converts a decimal state to its binary vector representation.
    bin_str = dec2bin(dec_val, num_bits);
    binary_vector = bin_str - '0'; % Convert char array '101' to numeric array [1 0 1]
end

function generate_rewiring_report(filename, gene_list, res_model, non_res_model, res_inf, non_res_inf)
    % Generates the final detailed text report.
    fid = fopen(filename, 'w');
    if fid == -1, error('Could not open file for writing: %s', filename); end

    fprintf(fid, '======================================================================\n');
    fprintf(fid, ' PBN Rewiring and Influence Analysis: Responder vs. Non-Responder\n');
    fprintf(fid, '======================================================================\n\n');
    fprintf(fid, 'Date Generated: %s\n\n', datestr(now));
    fprintf(fid, 'NOTE: Influence is calculated using the formal functional sensitivity definition.\n\n');

    % --- Section 1: Influence Analysis ---
    fprintf(fid, '-------------------------------------------------\n');
    fprintf(fid, '1. Global Gene Influence Analysis\n');
    fprintf(fid, '-------------------------------------------------\n');
    fprintf(fid, 'This section quantifies the total influence of each gene as a REGULATOR.\n');
    fprintf(fid, 'The score is the sum of influences across all its targets.\n\n');

    res_total_influence = sum(res_inf, 2);
    non_res_total_influence = sum(non_res_inf, 2);

    [sorted_res_inf, res_idx] = sort(res_total_influence, 'descend');
    [sorted_non_res_inf, non_res_idx] = sort(non_res_total_influence, 'descend');

    fprintf(fid, '%-25s | %-25s\n', 'Responder Model', 'Non-Responder Model');
    fprintf(fid, '------------------------------------------------------------\n');
    fprintf(fid, '%-15s | %-8s | %-15s | %-8s\n', 'Gene', 'Influence', 'Gene', 'Influence');
    fprintf(fid, '------------------------------------------------------------\n');
    for i = 1:length(gene_list)
        fprintf(fid, '%-15s | %-8.2f | %-15s | %-8.2f\n', ...
            gene_list{res_idx(i)}, sorted_res_inf(i), ...
            gene_list{non_res_idx(i)}, sorted_non_res_inf(i));
    end
    fprintf(fid, '\n\n');
    
    % --- Section 2: Detailed Logical Rewiring ---
    fprintf(fid, '-------------------------------------------------\n');
    fprintf(fid, '2. Detailed Logical and Topological Rewiring\n');
    fprintf(fid, '-------------------------------------------------\n');
    fprintf(fid, 'This section details the specific changes in the regulatory logic for each gene.\n\n');
    
    for i = 1:length(gene_list)
        target_gene = gene_list{i};
        fprintf(fid, '====== GENE: %s ======\n\n', upper(target_gene));
        
        res_node = res_model.nodes.(target_gene);
        non_res_node = non_res_model.nodes.(target_gene);
        
        fprintf(fid, '  -- Responder (Sensitive) Logic --\n');
        print_functions_to_file(fid, res_node);
        
        fprintf(fid, '  -- Non-Responder (Resistant) Logic --\n');
        print_functions_to_file(fid, non_res_node);
        
        fprintf(fid, '\n');
    end

    fclose(fid);
end

function print_functions_to_file(fid, node_data)
    % Helper to format and print the functions for a single gene.
    if ~isfield(node_data, 'functions') || isempty(node_data.functions)
        fprintf(fid, '    (No functions defined)\n\n');
        return;
    end
    
    for i = 1:length(node_data.functions)
        func = node_data.functions(i);
        func_str = strrep(func.function, '&', 'AND');
        func_str = strrep(func_str, '|', 'OR');
        func_str = strrep(func_str, '~', 'NOT ');
        fprintf(fid, '    - Prob: %.2f, Function: %s\n', func.probability, func_str);
    end
    fprintf(fid, '\n');
end
