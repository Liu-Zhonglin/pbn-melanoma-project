% =========================================================================
% MAIN SCRIPT: ANALYZE PBN ATTRACTORS (v4 - Parallel Computing)
% =========================================================================
% This version uses the Parallel Computing Toolbox (parfor) to drastically
% speed up the analysis by using all available CPU cores.
% =========================================================================

clear; clc;

% --- 0. Setup Environment ---
fprintf('--- Step 0: Setting up MATLAB environment ---\n');
toolbox_path = '/Users/liuzhonglin/Desktop/PBN_Melanoma_Project/Improved/Toolbox/pbn-matlab-toolbox';
addpath(genpath(toolbox_path));
fprintf('Added PBN toolbox to path: %s\n\n', toolbox_path);

% --- 1. Load and Convert the PBN Model ---
fprintf('--- Step 1: Loading and Converting PBN Model ---\n');
json_filename = 'non_responder_PBN_model_mi_loose.json';
json_data = jsondecode(fileread(json_filename));
model = convert_json_to_toolbox_format(json_data);
fprintf('Model conversion complete.\n  - Genes: %d\n  - Total Functions: %d\n\n', length(model.gene_list), sum(model.nf));

% --- 2. Enumerate All Deterministic BNs ---
fprintf('--- Step 2: Enumerating all possible deterministic BNs ---\n');
choice_indices = arrayfun(@(n) 1:n, model.nf, 'UniformOutput', false);
num_genes = length(model.nf);
grid_outputs = cell(1, num_genes);
[grid_outputs{:}] = ndgrid(choice_indices{:});
bn_combinations = cell2mat(cellfun(@(c) c(:), grid_outputs, 'UniformOutput', false));
total_bns = size(bn_combinations, 1);
fprintf('Generated %d unique Boolean Networks to analyze.\n\n', total_bns);

% --- 3. Main Analysis Loop (PARALLEL) ---
fprintf('--- Step 3: Starting PARALLEL attractor analysis for each BN ---\n');
fprintf('This may take a very long time, but will use all available CPU cores.\n');

% Start a parallel pool of workers (MATLAB does this automatically with parfor)
gcp; 

% We need a special way to handle the progress bar in a parfor loop
% This requires a helper object from the File Exchange or a simpler method.
% For simplicity, we will print progress less frequently.
% A more advanced solution is 'parfor_progress' from the File Exchange.

tic; % Start a timer for the whole loop

% We cannot use a standard containers.Map inside a parfor loop for accumulation.
% Instead, we will store results in a cell array and aggregate them after.
num_bns = size(bn_combinations, 1);
all_attractor_strings = cell(num_bns, 1);
all_bn_probabilities = zeros(num_bns, 1);

parfor i = 1:num_bns
    % Get the choices for the current BN
    current_bn_choices = bn_combinations(i, :);
    
    % Construct the specific BN
    [F_single, varF_single, nv_single] = get_bn_from_pbn(model, current_bn_choices);
    
    % Calculate its probability
    bn_probability = get_bn_probability(model, current_bn_choices);
    
    % Find attractors
    [~, Avec] = bnAsparse(F_single, varF_single, nv_single);
    [ab, ~] = bnAttractor(Avec);
    
    % Store the results for this BN
    num_attractors_found = length(unique(ab(ab < 0)));
    attractor_results_for_i = cell(num_attractors_found, 1);
    for k = 1:num_attractors_found
        attractor_states = find(ab == -k) - 1;
        attractor_results_for_i{k} = strjoin(string(sort(attractor_states)), ',');
    end
    
    % We store the results in a way that can be aggregated later
    all_attractor_strings{i} = attractor_results_for_i;
    all_bn_probabilities(i) = bn_probability;
end

loop_time = toc;
fprintf('Parallel loop finished in %s.\n\n', datestr(seconds(loop_time), 'HH:MM:SS'));

% --- 4. Aggregate Results (Post-Processing) ---
fprintf('--- Step 4: Aggregating results from parallel workers ---\n');
attractor_summary = containers.Map('KeyType', 'char', 'ValueType', 'double');

for i = 1:num_bns
    bn_prob = all_bn_probabilities(i);
    attractors_in_bn = all_attractor_strings{i};
    for j = 1:length(attractors_in_bn)
        canonical_str = attractors_in_bn{j};
        if isKey(attractor_summary, canonical_str)
            attractor_summary(canonical_str) = attractor_summary(canonical_str) + bn_prob;
        else
            attractor_summary(canonical_str) = bn_prob;
        end
    end
end

% --- 5. Display Final Report ---
fprintf('\n--- PBN ATTRACTOR ANALYSIS COMPLETE ---\n');
% (The rest of the display code is the same as before)
if isempty(attractor_summary)
    fprintf('No attractors were found.\n');
else
    keys = attractor_summary.keys;
    vals = cell2mat(attractor_summary.values);
    results_table = table('Size', [length(keys), 3], ...
        'VariableTypes', {'double', 'double', 'string'}, ...
        'VariableNames', {'Attractor_Size', 'Total_Probability', 'AttractorStates_Decimal'});
    for i = 1:length(keys)
        states_str = string(keys{i});
        results_table.AttractorStates_Decimal(i) = states_str;
        results_table.Total_Probability(i) = vals(i);
        results_table.Attractor_Size(i) = length(strsplit(states_str, ','));
    end
    results_table = sortrows(results_table, 'Total_Probability', 'descend');
    fprintf('The following attractors were identified across the PBN, ranked by probability:\n\n');
    disp(results_table);
    fprintf('\n--- CONCLUSION ---\n');
    fprintf('This table shows the stable states (attractors) of the entire system.\n');
end