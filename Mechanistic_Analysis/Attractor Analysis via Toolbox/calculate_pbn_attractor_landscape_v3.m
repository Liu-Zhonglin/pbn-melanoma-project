function results_table = calculate_pbn_attractor_landscape_v3(json_filename, num_samples)
% =========================================================================
% FUNCTION: calculate_pbn_attractor_landscape_v3
%
% PURPOSE:
%   Calculates the attractor landscape for a PBN model.
%
% v3 REVISIONS:
% - FIX: Moved the nested function definition for the progress bar
%   outside of the try...catch block to resolve the MATLAB parsing error.
% =========================================================================

    % --- 1. Load and Convert Model ---
    fprintf('    -> Loading and converting model: %s\n', json_filename);
    json_data = jsondecode(fileread(json_filename));
    model = convert_json_to_toolbox_format(json_data);

    % --- 2. Generate BN Samples ---
    fprintf('    -> Generating %d random BN samples...\n', num_samples);
    num_genes = length(model.gene_list);
    bn_choices_to_sample = zeros(num_samples, num_genes);
    for i = 1:num_genes
        probabilities = model.cij(1:model.nf(i), i);
        num_functions = model.nf(i);
        bn_choices_to_sample(:, i) = randsample(1:num_functions, num_samples, true, probabilities);
    end

    % --- 3. Run Parallel Analysis with Progress Bar ---
    fprintf('    -> Starting parallel analysis of %d networks...\n', num_samples);
    
    % --- START: Progress Bar Setup (CORRECTED) ---
    progress_queue = []; % Initialize queue as empty
    progress_counter = 0;
    total_items = num_samples;
    progress_msg_len = 0;

    % DEFINE the nested function in the main function's scope, NOT in a conditional block.
    function update_progress_bar(~)
        progress_counter = progress_counter + 1;
        percent_done = 100 * progress_counter / total_items;
        fprintf(repmat('\b', 1, progress_msg_len)); % Erase previous line
        progress_msg = sprintf('       Progress: %.1f%% (%d / %d)', percent_done, progress_counter, total_items);
        fprintf('%s', progress_msg);
        progress_msg_len = numel(progress_msg);
    end

    % Now, USE the function handle inside the try...catch block.
    try
        progress_queue = parallel.pool.DataQueue;
        afterEach(progress_queue, @update_progress_bar);
    catch
        fprintf('       (Progress bar disabled.)\n');
    end
    % --- END: Progress Bar Setup ---
    
    tic;
    all_attractor_strings = cell(num_samples, 1);
    
    parfor i = 1:num_samples
        current_bn_choices = bn_choices_to_sample(i, :);
        [F_single, varF_single] = pbnSelect(model, current_bn_choices);
        nv_single = model.nv(cumsum([1, model.nf(1:end-1)]));
        [~, Avec] = bnAsparse(F_single, varF_single, nv_single);
        [ab, ~] = bnAttractor(Avec);
        
        num_attractors_found = length(unique(ab(ab < 0)));
        attractor_results_for_i = cell(num_attractors_found, 1);
        for k = 1:num_attractors_found
            attractor_states = find(ab == -k) - 1;
            attractor_results_for_i{k} = strjoin(string(sort(attractor_states)), ',');
        end
        all_attractor_strings{i} = attractor_results_for_i;
        
        if ~isempty(progress_queue)
            send(progress_queue, 1);
        end
    end
    
    if ~isempty(progress_queue), fprintf('\n'); end
    loop_time = toc;
    fprintf('    -> Parallel loop finished in %.2f seconds.\n', loop_time);

    % --- 4. Aggregate and Format Results ---
    attractor_summary = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for i = 1:num_samples
        attractors_in_sample = all_attractor_strings{i};
        for j = 1:length(attractors_in_sample)
            canonical_str = attractors_in_sample{j};
            if isKey(attractor_summary, canonical_str)
                attractor_summary(canonical_str) = attractor_summary(canonical_str) + 1;
            else
                attractor_summary(canonical_str) = 1;
            end
        end
    end

    if isempty(attractor_summary)
        results_table = table();
        return;
    end

    keys = attractor_summary.keys;
    counts = cell2mat(attractor_summary.values);
    vals = counts / num_samples; 
    
    results_table = table('Size', [length(keys), 4], 'VariableTypes', {'double', 'double', 'double', 'string'}, 'VariableNames', {'Attractor_Size', 'Frequency_Count', 'Estimated_Probability', 'AttractorStates_Decimal'});
    for i = 1:length(keys)
        states_str = string(keys{i});
        results_table.AttractorStates_Decimal(i) = states_str;
        results_table.Estimated_Probability(i) = vals(i);
        results_table.Frequency_Count(i) = counts(i);
        results_table.Attractor_Size(i) = length(strsplit(states_str, ','));
    end
    results_table = sortrows(results_table, 'Estimated_Probability', 'descend');
end