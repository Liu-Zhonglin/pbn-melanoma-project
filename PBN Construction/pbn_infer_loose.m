% =========================================================================
% SCRIPT: pbn_infer_loose.m (Complete and Corrected Version)
%
% PURPOSE:
% Infers PBN models with adjustable model selection stringency. This is the
% full, self-contained script with all required helper functions included
% to prevent parallel computing errors.
%
% vLoose REVISIONS:
% - Introduced 'MODEL_SELECTION_CRITERION' to switch between 'AIC' (strict)
%   and 'MI' (loose) inference.
% - Widened TOP_X_DYN_POOL to 15 to expand the search space.
% =========================================================================

fprintf('====== STARTING PBN INFERENCE (LOOSE CRITERIA) ======\n');

%% --- 1. CONFIGURATION ---
conditions = {'responder', 'non_responder'};
ranking_file = 'dynGENIE3_ranked_interactions.csv';
binarized_data_folder = './';

% --- User-Defined Parameters ---
TOP_X_DYN_POOL = 10;
MAX_K_TO_TEST = 4;
NUM_BEST_FUNCS = 4;

% --- CHOOSE YOUR MODEL SELECTION METHOD ---
% 'AIC': Strict, penalizes complexity to prevent overfitting.
% 'MI':  Loose, selects regulators that maximize information.
MODEL_SELECTION_CRITERION = 'MI'; 

fprintf('Model Selection Criterion set to: %s\n', MODEL_SELECTION_CRITERION);

%% --- 2. DATA LOADING & INFERENCE LOOP ---
try
    ranking_table = readtable(ranking_file);
catch e
    fprintf('ERROR: Could not load Ranking file: %s\n', ranking_file);
    rethrow(e);
end
fprintf('Loaded Rankings successfully from: %s\n', ranking_file);

if isempty(gcp('nocreate')), parpool; end

for i = 1:length(conditions)
    condition_name = conditions{i};
    bdata_file = fullfile(binarized_data_folder, sprintf('binarized_%s_final.csv', condition_name));
    output_json_file = sprintf('%s_PBN_model_mi_loose.json', condition_name);

    fprintf('\n\n====== INFERRING MODEL FOR: %s ======\n', upper(condition_name));
    
    PBN_model_map = infer_pbn_with_criterion(bdata_file, ranking_table, TOP_X_DYN_POOL, MAX_K_TO_TEST, NUM_BEST_FUNCS, MODEL_SELECTION_CRITERION);

    fprintf('\nConverting PBN model to JSON format...\n');
    json_output_struct = convert_pbn_map_to_json_struct_mi(PBN_model_map);
    json_text = jsonencode(json_output_struct, 'PrettyPrint', true);
    
    fid = fopen(output_json_file, 'w');
    if fid == -1, error('Cannot create JSON file: %s', output_json_file); end
    fprintf(fid, '%s', json_text);
    fclose(fid);
    
    fprintf('Final PBN model for "%s" condition saved to: %s\n', condition_name, output_json_file);
end

fprintf('\n\n====== FINAL PBN INFERENCE COMPLETE! ======\n');


%% --- PRIMARY HELPER FUNCTION (MODIFIED FOR CRITERION) ---
function PBN_model = infer_pbn_with_criterion(bdata_file, ranking_table, top_x_pool, max_k_test, num_best_funcs, criterion)
    binarized_table = readtable(bdata_file, 'ReadRowNames', true);
    target_genes = binarized_table.Properties.RowNames;
    num_targets = length(target_genes);
    results_cell = cell(num_targets, 1);
    
    fprintf('Inferring PBN for %d target genes...\n', num_targets);
    
    parfor i = 1:num_targets
        result_struct = struct();
        targetGene = target_genes{i};
        fprintf('Processing gene: %s\n', targetGene);
        
        % --- STAGE 1: Select optimal number of regulators (k) ---
        best_aic = inf;
        max_mi_for_k = -inf;
        optimal_k = 0;
        
        potential_regs_full = ranking_table.regulatory_gene(strcmp(ranking_table.target_gene, targetGene));
        potential_regs_full = intersect(potential_regs_full, binarized_table.Properties.RowNames, 'stable');
        
        if length(potential_regs_full) > top_x_pool, potential_regs_pool = potential_regs_full(1:top_x_pool);
        else, potential_regs_pool = potential_regs_full; end

        for k = 1:min(max_k_test, length(potential_regs_pool))
            regulators_to_test = potential_regs_pool(1:k);
            
            if strcmpi(criterion, 'AIC')
                [~, ~, min_error] = find_best_function_for_set(targetGene, regulators_to_test, binarized_table);
                if isinf(min_error), continue; end
                num_samples = width(binarized_table);
                aic_score = 2*k + num_samples * log(min_error + 1e-9); 
                if aic_score < best_aic, best_aic = aic_score; optimal_k = k; end
            
            elseif strcmpi(criterion, 'MI')
                [~, mi_scores] = get_all_functions_and_mi(targetGene, regulators_to_test, binarized_table);
                current_max_mi = max(mi_scores);
                if current_max_mi > max_mi_for_k
                    max_mi_for_k = current_max_mi;
                    optimal_k = k;
                end
            end
        end
        
        if optimal_k == 0, results_cell{i} = struct(); continue; end
        final_regulators = potential_regs_pool(1:optimal_k);
        
        % --- STAGE 2: Score all functions using Mutual Information (MI) ---
        [all_funcs, all_mi_scores] = get_all_functions_and_mi(targetGene, final_regulators, binarized_table);
        [sorted_mi, sort_idx] = sort(all_mi_scores, 'descend');
        sorted_funcs = all_funcs(sort_idx);
        
        num_to_keep = min(num_best_funcs, length(sorted_funcs));
        top_funcs = sorted_funcs(1:num_to_keep);
        top_mi = sorted_mi(1:num_to_keep);
        
        mi_sum = sum(top_mi);
        if mi_sum <= 0, probabilities = ones(size(top_mi)) / length(top_mi);
        else, probabilities = top_mi / mi_sum; end
        
        result_struct.Regulators = final_regulators;
        result_struct.BestFunctions = top_funcs;
        result_struct.Probabilities = probabilities';
        result_struct.MI = top_mi';
        results_cell{i} = result_struct;
    end
    
    PBN_model = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for i = 1:num_targets
        if ~isempty(fieldnames(results_cell{i})), PBN_model(target_genes{i}) = results_cell{i}; end
    end
end


%% --- FULL SET OF UNMODIFIED HELPER FUNCTIONS ---
% These are now included in this file to ensure the parallel workers can find them.

function [all_funcs, all_mi_scores] = get_all_functions_and_mi(target, regs, data)
    k = length(regs);
    observed_output = data{target, :};
    num_samples = width(data);
    
    if k == 0 || num_samples == 0, all_funcs={}; all_mi_scores=[]; return; end
    
    input_data = data{regs, :};
    num_functions = 2^(2^k);
    all_funcs = cell(num_functions, 1);
    all_mi_scores = zeros(num_functions, 1);
    input_as_decimal = bi2de(input_data', 'left-msb');
    
    p_obs_1 = sum(observed_output) / num_samples;
    p_obs_0 = 1 - p_obs_1;
    if p_obs_0 == 0 || p_obs_1 == 0, H_obs = 0;
    else, H_obs = - (p_obs_0 * log2(p_obs_0) + p_obs_1 * log2(p_obs_1)); end
    
    for i = 0:(num_functions-1)
        truth_table = de2bi(i, 2^k, 'left-msb');
        all_funcs{i+1} = truth_table;
        predicted_output = truth_table(input_as_decimal + 1);
        
        p_pred_1 = sum(predicted_output) / num_samples;
        p_pred_0 = 1 - p_pred_1;
        
        p_joint_00 = sum(observed_output == 0 & predicted_output == 0) / num_samples;
        p_joint_01 = sum(observed_output == 0 & predicted_output == 1) / num_samples;
        p_joint_10 = sum(observed_output == 1 & predicted_output == 0) / num_samples;
        p_joint_11 = sum(observed_output == 1 & predicted_output == 1) / num_samples;

        H_cond = 0;
        if p_pred_0 > 0
            p_cond_0_given_0 = p_joint_00 / p_pred_0;
            p_cond_1_given_0 = p_joint_10 / p_pred_0;
            if p_cond_0_given_0 > 0, H_cond = H_cond - p_joint_00 * log2(p_cond_0_given_0); end
            if p_cond_1_given_0 > 0, H_cond = H_cond - p_joint_10 * log2(p_cond_1_given_0); end
        end
        if p_pred_1 > 0
            p_cond_0_given_1 = p_joint_01 / p_pred_1;
            p_cond_1_given_1 = p_joint_11 / p_pred_1;
            if p_cond_0_given_1 > 0, H_cond = H_cond - p_joint_01 * log2(p_cond_0_given_1); end
            if p_cond_1_given_1 > 0, H_cond = H_cond - p_joint_11 * log2(p_cond_1_given_1); end
        end
        
        mi = H_obs - H_cond;
        
        if H_obs == 0, all_mi_scores(i+1) = 1;
        else, all_mi_scores(i+1) = mi / H_obs; end
    end
end

function json_struct = convert_pbn_map_to_json_struct_mi(pbn_map)
    gene_names = keys(pbn_map);
    nodes_struct = struct();
    for i = 1:length(gene_names)
        gene_name = gene_names{i};
        gene_info = pbn_map(gene_name);
        functions_array = [];
        if isfield(gene_info, 'BestFunctions')
            for f_idx = 1:length(gene_info.BestFunctions)
                func_struct = struct();
                func_struct.function = truth_table_to_formula(gene_info.BestFunctions{f_idx}, gene_info.Regulators);
                func_struct.inputs = gene_info.Regulators;
                func_struct.mi = gene_info.MI(f_idx);
                func_struct.probability = gene_info.Probabilities(f_idx);
                if isempty(functions_array), functions_array = func_struct;
                else, functions_array(end+1) = func_struct; end
            end
        end
        nodes_struct.(gene_name).functions = functions_array;
    end
    json_struct.nodes = nodes_struct;
end

function [best_func, max_cod, min_error] = find_best_function_for_set(target, regs, data)
    k = length(regs); observed_output = data{target, :}; num_samples = width(data);
    error_baseline = min([sum(observed_output), num_samples - sum(observed_output)]);
    if error_baseline == 0, best_func = {observed_output}; max_cod = 1.0; min_error = 0; return;
    elseif k == 0, best_func = {round(mean(observed_output))}; min_error = sum(abs(best_func{1} - observed_output)); max_cod = 1 - (min_error / error_baseline); return; end
    input_data = data{regs, :}; min_error = num_samples; max_cod = -inf; best_func = {};
    input_as_decimal = bi2de(input_data', 'left-msb');
    for i = 0:(2^(2^k)-1)
        truth_table = de2bi(i, 2^k, 'left-msb'); predicted_output = truth_table(input_as_decimal + 1);
        error_func = sum(abs(predicted_output - observed_output));
        if error_func < min_error, min_error = error_func; max_cod = 1 - (error_func / error_baseline); best_func = {truth_table}; end
    end
end

function formula = truth_table_to_formula(truth_table, regulator_names)
    num_inputs = length(regulator_names); if num_inputs == 0, formula = num2str(truth_table(1)); return; end
    minterms = {};
    for i = 0:(2^num_inputs - 1)
        if truth_table(i+1) == 1
            binary_state = de2bi(i, num_inputs, 'left-msb'); term_parts = {};
            for j = 1:num_inputs
                if binary_state(j) == 1, term_parts{end+1} = regulator_names{j};
                else, term_parts{end+1} = sprintf('~%s', regulator_names{j}); end
            end
            minterms{end+1} = strjoin(term_parts, ' & ');
        end
    end
    if isempty(minterms), formula = '0'; else, formula = strjoin(minterms, ' | '); end
end